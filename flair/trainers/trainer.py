import copy
import datetime
import inspect
import logging
import os
import sys
import time
import warnings
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch.optim.sgd import SGD
from torch.utils.data.dataset import ConcatDataset

from flair.nn import Model

try:
    from apex import amp
except ImportError:
    amp = None

import random

from torch.optim.lr_scheduler import OneCycleLR  # type: ignore

import flair
import flair.nn
from flair.data import Corpus, Dictionary, MultiCorpus, _len_dataset
from flair.datasets import DataLoader
from flair.optim import ExpAnnealLR, LinearSchedulerWithWarmup
from flair.training_utils import (
    AnnealOnPlateau,
    WeightExtractor,
    add_file_handler,
    identify_dynamic_embeddings,
    init_output_file,
    log_line,
    store_embeddings,
)

log = logging.getLogger("flair")


class ModelTrainer:
    def __init__(
        self,
        model: flair.nn.Model,
        corpus: Corpus,
    ):
        """
        Initialize a model trainer
        :param model: The model that you want to train. The model should inherit from flair.nn.Model  # noqa: E501
        :param corpus: The dataset used to train the model, should be of type Corpus
        """
        self.model: flair.nn.Model = model
        self.corpus: Corpus = corpus

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
        eval_on_train_fraction: float = 0.0,
        eval_on_train_shuffle: bool = False,
        save_model_each_k_epochs: int = 0,
        tensorboard_comment: str = "",
        use_swa: bool = False,
        use_final_model_for_eval: bool = False,
        gold_label_dictionary_for_eval: Optional[Dictionary] = None,
        create_file_logs: bool = True,
        create_loss_file: bool = True,
        epoch: int = 0,
        use_tensorboard: bool = False,
        tensorboard_log_dir=None,
        metrics_for_tensorboard=[],
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        scheduler_state_dict: Optional[Dict[str, Any]] = None,
        save_optimizer_state: bool = False,
        **kwargs,
    ) -> dict:
        """
        Trains any class that implements the flair.nn.Model interface.
        :param base_path: Main path to which all output during training is logged and models are saved  # noqa: E501
        :param learning_rate: Initial learning rate (or max, if scheduler is OneCycleLR)  # noqa: E501
        :param mini_batch_size: Size of mini-batches during training
        :param mini_batch_chunk_size: If mini-batches are larger than this number, they get broken down into chunks of this size for processing purposes  # noqa: E501
        :param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.  # noqa: E501
        :param scheduler: The learning rate scheduler to use
        :param checkpoint: If True, a full checkpoint is saved at end of each epoch  # noqa: E501
        :param cycle_momentum: If scheduler is OneCycleLR, whether the scheduler should cycle also the momentum  # noqa: E501
        :param anneal_factor: The factor by which the learning rate is annealed
        :param patience: Patience is the number of epochs with no improvement the Trainer waits  # noqa: E501
         until annealing the learning rate
        :param min_learning_rate: If the (in multi lr case: all) learning rate falls below this threshold, training terminates  # noqa: E501
        :param warmup_fraction: Fraction of warmup steps if the scheduler is LinearSchedulerWithWarmup  # noqa: E501
        :param train_with_dev:  If True, the data from dev split is added to the training data  # noqa: E501
        :param train_with_test: If True, the data from test split is added to the training data  # noqa: E501
        :param monitor_train: If True, training data is evaluated at end of each epoch
        :param monitor_test: If True, test data is evaluated at end of each epoch
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),  # noqa: E501
        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param save_final_model: If True, final model is saved
        :param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate  # noqa: E501
        :param shuffle: If True, data is shuffled during training
        :param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing  # noqa: E501
        parameter selection.
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
        :param use_tensorboard: If True, writes out tensorboard information
        :param tensorboard_log_dir: Directory into which tensorboard log files will be written  # noqa: E501
        :param metrics_for_tensorboard: List of tuples that specify which metrics (in addition to the main_score) shall be plotted in tensorboard, could be [("macro avg", 'f1-score'), ("macro avg", 'precision')] for example  # noqa: E501
        :param kwargs: Other arguments for the Optimizer
        :return:
        """

        # create a model card for this model with Flair and PyTorch version
        model_card: Dict[str, Any] = {
            "flair_version": flair.__version__,
            "pytorch_version": torch.__version__,
        }

        # also record Transformers version if library is loaded
        try:
            import transformers

            model_card["transformers_version"] = transformers.__version__
        except ImportError:
            pass

        # remember all parameters used in train() call
        local_variables = locals()
        training_parameters = {}
        for parameter in signature(self.train).parameters:
            training_parameters[parameter] = local_variables[parameter]
        model_card["training_parameters"] = training_parameters

        # add model card to model
        self.model.model_card = model_card
        assert self.corpus.train
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                if tensorboard_log_dir is not None and not os.path.exists(tensorboard_log_dir):
                    os.mkdir(tensorboard_log_dir)
                writer = SummaryWriter(log_dir=tensorboard_log_dir, comment=tensorboard_comment)
                log.info(f"tensorboard logging path is {tensorboard_log_dir}")

            except ImportError:
                log_line(log)
                log.warning("ATTENTION! PyTorch >= 1.1.0 and pillow are required" "for TensorBoard support!")
                log_line(log)
                use_tensorboard = False
                pass

        if use_amp:
            if sys.version_info < (3, 0):
                raise RuntimeError("Apex currently only supports Python 3. Aborting.")
            if amp is None:
                raise RuntimeError(
                    "Failed to import apex. Please install apex from "
                    "https://www.github.com/nvidia/apex "
                    "to enable mixed-precision training."
                )

        if not eval_batch_size:
            eval_batch_size = mini_batch_size
        if mini_batch_chunk_size is None:
            mini_batch_chunk_size = mini_batch_size

        if inspect.isclass(optimizer):
            # if optimizer is class, trainer will create a single parameter group
            initial_learning_rate = [learning_rate]
        else:
            initial_learning_rate = [group["lr"] for group in optimizer.param_groups]

        if not isinstance(min_learning_rate, list):
            min_learning_rate = [min_learning_rate] * len(initial_learning_rate)

        for i, lr in enumerate(initial_learning_rate):
            if lr < min_learning_rate[i]:
                min_learning_rate[i] = lr / 10

        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True, parents=True)

        self.check_for_and_delete_previous_best_models(base_path)

        # determine what splits (train, dev, test) to evaluate and log
        log_train = True if monitor_train else False
        log_test = True if (not param_selection_mode and self.corpus.test and monitor_test) else False
        log_dev = False if train_with_dev or not self.corpus.dev else True
        log_train_part = True if (eval_on_train_fraction == "dev" or eval_on_train_fraction > 0.0) else False

        if log_train_part:
            train_part_size = (
                _len_dataset(self.corpus.dev)
                if eval_on_train_fraction == "dev"
                else int(_len_dataset(self.corpus.train) * eval_on_train_fraction)
            )

            assert train_part_size > 0
            if not eval_on_train_shuffle:
                train_part_indices = list(range(train_part_size))
                train_part = torch.utils.data.dataset.Subset(self.corpus.train, train_part_indices)

        # prepare loss logging file and set up header
        loss_txt = init_output_file(base_path, "loss.tsv") if create_loss_file else None

        weight_extractor = WeightExtractor(base_path)

        # if optimizer class is passed, instantiate:
        if inspect.isclass(optimizer):
            kwargs["lr"] = learning_rate
            optimizer = optimizer(self.model.parameters(), **kwargs)

        if use_swa:
            import torchcontrib

            optimizer = torchcontrib.optim.SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=learning_rate)

        # from here on, use list of learning rates
        current_learning_rate: List = [group["lr"] for group in optimizer.param_groups]

        if use_amp:
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=amp_opt_level)

        optimizer = cast(torch.optim.Optimizer, optimizer)

        # load existing optimizer state dictionary if it exists
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        # minimize training loss if training with dev data, else maximize dev score
        anneal_mode = "min" if train_with_dev or anneal_against_dev_loss else "max"
        best_validation_score = 100000000000 if train_with_dev or anneal_against_dev_loss else 0.0

        dataset_size = _len_dataset(self.corpus.train)
        if train_with_dev:
            dataset_size += _len_dataset(self.corpus.dev)

        # if scheduler is passed as a class, instantiate
        if inspect.isclass(scheduler):
            if scheduler == OneCycleLR:
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=current_learning_rate,
                    steps_per_epoch=dataset_size // mini_batch_size + 1,
                    epochs=max_epochs - epoch,
                    # if we load a checkpoint, we have already trained for epoch
                    pct_start=0.0,
                    cycle_momentum=cycle_momentum,
                )
            elif scheduler == LinearSchedulerWithWarmup:
                steps_per_epoch = (dataset_size + mini_batch_size - 1) / mini_batch_size
                num_train_steps = int(steps_per_epoch * max_epochs)
                num_warmup_steps = int(num_train_steps * warmup_fraction)

                scheduler = LinearSchedulerWithWarmup(
                    optimizer,
                    num_train_steps=num_train_steps,
                    num_warmup_steps=num_warmup_steps,
                )
            else:
                scheduler = scheduler(
                    optimizer,
                    factor=anneal_factor,
                    patience=patience,
                    initial_extra_patience=initial_extra_patience,
                    mode=anneal_mode,
                    verbose=True,
                )

        # load existing scheduler state dictionary if it exists
        if scheduler_state_dict:
            scheduler.load_state_dict(scheduler_state_dict)

        # update optimizer and scheduler in model card
        model_card["training_parameters"]["optimizer"] = optimizer
        model_card["training_parameters"]["scheduler"] = scheduler

        if isinstance(scheduler, OneCycleLR) and batch_growth_annealing:
            raise ValueError("Batch growth with OneCycle policy is not implemented.")

        train_data = self.corpus.train

        # if training also uses dev/train data, include in training set
        if train_with_dev or train_with_test:

            parts = [self.corpus.train]
            if train_with_dev and self.corpus.dev:
                parts.append(self.corpus.dev)
            if train_with_test and self.corpus.test:
                parts.append(self.corpus.test)

            train_data = ConcatDataset(parts)

        # initialize sampler if provided
        if sampler is not None:
            # init with default values if only class is provided
            if inspect.isclass(sampler):
                sampler = sampler()
            # set dataset to sample from
            sampler.set_dataset(train_data)
            shuffle = False

        dev_score_history = []
        dev_loss_history = []
        train_loss_history = []

        micro_batch_size = mini_batch_chunk_size

        # this field stores the names of all dynamic embeddings in the model (determined after first forward pass)
        dynamic_embeddings = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            if create_file_logs:
                log_handler = add_file_handler(log, base_path / "training.log")
            else:
                log_handler = None

            lr_info = ",".join([f"{lr:.6f}" for lr in current_learning_rate])

            log_line(log)
            log.info(f'Model: "{self.model}"')
            log_line(log)
            log.info(f'Corpus: "{self.corpus}"')
            log_line(log)
            log.info("Parameters:")
            log.info(f' - learning_rate: "{lr_info}"')
            log.info(f' - mini_batch_size: "{mini_batch_size}"')
            log.info(f' - patience: "{patience}"')
            log.info(f' - anneal_factor: "{anneal_factor}"')
            log.info(f' - max_epochs: "{max_epochs}"')
            log.info(f' - shuffle: "{shuffle}"')
            log.info(f' - train_with_dev: "{train_with_dev}"')
            log.info(f' - batch_growth_annealing: "{batch_growth_annealing}"')
            log_line(log)
            log.info(f'Model training base path: "{base_path}"')
            log_line(log)
            log.info(f"Device: {flair.device}")
            log_line(log)
            log.info(f"Embeddings storage mode: {embeddings_storage_mode}")

            previous_learning_rate = current_learning_rate

            momentum = [group["momentum"] if "momentum" in group else 0 for group in optimizer.param_groups]

            for epoch in range(epoch + 1, max_epochs + 1):
                log_line(log)

                # update epoch in model card
                model_card["training_parameters"]["epoch"] = epoch

                if anneal_with_prestarts:
                    last_epoch_model_state_dict = copy.deepcopy(self.model.state_dict())

                if eval_on_train_shuffle:
                    train_part_indices = list(range(_len_dataset(self.corpus.train)))
                    random.shuffle(train_part_indices)
                    train_part_indices = train_part_indices[:train_part_size]
                    train_part = torch.utils.data.dataset.Subset(self.corpus.train, train_part_indices)

                # get new learning rate
                current_learning_rate = [group["lr"] for group in optimizer.param_groups]

                lr_changed = any([lr != prev_lr for lr, prev_lr in zip(current_learning_rate, previous_learning_rate)])

                if lr_changed and batch_growth_annealing:
                    mini_batch_size *= 2

                # reload last best model if annealing with restarts is enabled
                if (
                    (anneal_with_restarts or anneal_with_prestarts)
                    and lr_changed
                    and os.path.exists(base_path / "best-model.pt")
                ):
                    if anneal_with_restarts:
                        log.info("resetting to best model")
                        self.model.load_state_dict(self.model.load(base_path / "best-model.pt").state_dict())
                    if anneal_with_prestarts:
                        log.info("resetting to pre-best model")
                        self.model.load_state_dict(self.model.load(base_path / "pre-best-model.pt").state_dict())

                previous_learning_rate = current_learning_rate
                if use_tensorboard:
                    if len(current_learning_rate) == 1:
                        writer.add_scalar("learning_rate", current_learning_rate[0], epoch)
                    else:
                        for i, lr in enumerate(current_learning_rate):
                            writer.add_scalar(f"learning_rate_{i}", lr, epoch)

                all_lrs_too_small = all([lr < min_lr for lr, min_lr in zip(current_learning_rate, min_learning_rate)])

                # stop training if learning rate becomes too small
                if not isinstance(scheduler, (OneCycleLR, LinearSchedulerWithWarmup)) and all_lrs_too_small:
                    log_line(log)
                    log.info("learning rate too small - quitting training!")
                    log_line(log)
                    break

                batch_loader = DataLoader(
                    train_data,
                    batch_size=mini_batch_size,
                    shuffle=shuffle if epoch > 1 else False,  # never shuffle the first epoch
                    num_workers=0 if num_workers is None else num_workers,
                    sampler=sampler,
                )

                self.model.train()

                train_loss: float = 0

                seen_batches = 0
                total_number_of_batches = len(batch_loader)

                modulo = max(1, int(total_number_of_batches / 10))

                # process mini-batches
                batch_time = 0.0
                average_over = 0
                for batch_no, batch in enumerate(batch_loader):

                    start_time = time.time()

                    # zero the gradients on the model and optimizer
                    self.model.zero_grad()
                    optimizer.zero_grad()

                    # if necessary, make batch_steps
                    batch_steps = [batch]
                    if len(batch) > micro_batch_size:
                        batch_steps = [batch[x : x + micro_batch_size] for x in range(0, len(batch), micro_batch_size)]

                    # forward and backward for batch
                    for batch_step in batch_steps:

                        # forward pass
                        loss = self.model.forward_loss(batch_step)

                        if isinstance(loss, tuple):
                            average_over += loss[1]
                            loss = loss[0]

                        # Backward
                        if use_amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        train_loss += loss.item()

                        # identify dynamic embeddings (always deleted) on first sentence
                        if not dynamic_embeddings:
                            dynamic_embeddings = identify_dynamic_embeddings(batch[0])

                        # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                        store_embeddings(batch, embeddings_storage_mode, dynamic_embeddings)

                    # do the optimizer step
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    # do the scheduler step if one-cycle or linear decay
                    if isinstance(scheduler, (OneCycleLR, LinearSchedulerWithWarmup)):
                        scheduler.step()
                        # get new learning rate
                        current_learning_rate = [group["lr"] for group in optimizer.param_groups]

                        momentum = [
                            group["betas"][0] if "betas" in group else group.get("momentum", 0)
                            for group in optimizer.param_groups
                        ]

                    seen_batches += 1

                    batch_time += time.time() - start_time
                    if seen_batches % modulo == 0:
                        momentum_info = ""
                        if cycle_momentum:
                            momentum_info = " - momentum:" + ",".join([f"{m:.4f}" for m in momentum])

                        lr_info = ",".join([f"{lr:.6f}" for lr in current_learning_rate])

                        intermittent_loss = train_loss / average_over if average_over > 0 else train_loss / seen_batches

                        log.info(
                            f"epoch {epoch} - iter {seen_batches}/"
                            f"{total_number_of_batches} - loss "
                            f"{intermittent_loss:.8f} - samples/sec:"
                            f" {mini_batch_size * modulo / batch_time:.2f}"
                            f" - lr: {lr_info}{momentum_info}"
                        )
                        batch_time = 0.0
                        iteration = epoch * total_number_of_batches + batch_no
                        if not param_selection_mode and write_weights:
                            weight_extractor.extract_weights(self.model.state_dict(), iteration)

                if average_over != 0:
                    train_loss /= average_over

                self.model.eval()

                if save_model_each_k_epochs > 0 and epoch % save_model_each_k_epochs == 0:
                    log.info("saving model of current epoch")
                    model_name = "model_epoch_" + str(epoch) + ".pt"
                    self.model.save(base_path / model_name, checkpoint=save_optimizer_state)

                log_line(log)
                log.info(f"EPOCH {epoch} done: loss {train_loss:.4f} - lr {lr_info}")

                if use_tensorboard:
                    writer.add_scalar("train_loss", train_loss, epoch)

                # evaluate on train / dev / test split depending on training settings
                result_line: str = ""

                if log_train:
                    train_eval_result = self.model.evaluate(
                        self.corpus.train,
                        gold_label_type=self.model.label_type,
                        mini_batch_size=eval_batch_size,
                        num_workers=num_workers,
                        embedding_storage_mode=embeddings_storage_mode,
                        main_evaluation_metric=main_evaluation_metric,
                        gold_label_dictionary=gold_label_dictionary_for_eval,
                    )
                    result_line += f"\t{train_eval_result.log_line}"

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.train, embeddings_storage_mode, dynamic_embeddings)

                if log_train_part:
                    train_part_eval_result = self.model.evaluate(
                        train_part,
                        gold_label_type=self.model.label_type,
                        mini_batch_size=eval_batch_size,
                        num_workers=num_workers,
                        embedding_storage_mode=embeddings_storage_mode,
                        main_evaluation_metric=main_evaluation_metric,
                        gold_label_dictionary=gold_label_dictionary_for_eval,
                    )
                    result_line += f"\t{train_part_eval_result.loss}" f"\t{train_part_eval_result.log_line}"

                    log.info(
                        f"TRAIN_SPLIT : loss {train_part_eval_result.loss}"
                        f" - {main_evaluation_metric[1]}"
                        f" ({main_evaluation_metric[0]})"
                        f" {round(train_part_eval_result.main_score, 4)}"
                    )
                if use_tensorboard:
                    for (metric_class_avg_type, metric_type) in metrics_for_tensorboard:
                        writer.add_scalar(
                            f"train_{metric_class_avg_type}_{metric_type}",
                            train_part_eval_result.classification_report[metric_class_avg_type][metric_type],
                            epoch,
                        )

                if log_dev:
                    assert self.corpus.dev
                    dev_eval_result = self.model.evaluate(
                        self.corpus.dev,
                        gold_label_type=self.model.label_type,
                        mini_batch_size=eval_batch_size,
                        num_workers=num_workers,
                        out_path=base_path / "dev.tsv",
                        embedding_storage_mode=embeddings_storage_mode,
                        main_evaluation_metric=main_evaluation_metric,
                        gold_label_dictionary=gold_label_dictionary_for_eval,
                    )
                    result_line += f"\t{dev_eval_result.loss}\t{dev_eval_result.log_line}"
                    log.info(
                        f"DEV : loss {dev_eval_result.loss}"
                        f" - {main_evaluation_metric[1]}"
                        f" ({main_evaluation_metric[0]})"
                        f"  {round(dev_eval_result.main_score, 4)}"
                    )
                    # calculate scores using dev data if available
                    # append dev score to score history
                    dev_score_history.append(dev_eval_result.main_score)
                    dev_loss_history.append(dev_eval_result.loss)

                    dev_score = dev_eval_result.main_score

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.dev, embeddings_storage_mode, dynamic_embeddings)

                    if use_tensorboard:
                        writer.add_scalar("dev_loss", dev_eval_result.loss, epoch)
                        writer.add_scalar("dev_score", dev_eval_result.main_score, epoch)
                        for (
                            metric_class_avg_type,
                            metric_type,
                        ) in metrics_for_tensorboard:
                            writer.add_scalar(
                                f"dev_{metric_class_avg_type}_{metric_type}",
                                dev_eval_result.classification_report[metric_class_avg_type][metric_type],
                                epoch,
                            )

                if log_test:
                    assert self.corpus.test
                    test_eval_result = self.model.evaluate(
                        self.corpus.test,
                        gold_label_type=self.model.label_type,
                        mini_batch_size=eval_batch_size,
                        num_workers=num_workers,
                        out_path=base_path / "test.tsv",
                        embedding_storage_mode=embeddings_storage_mode,
                        main_evaluation_metric=main_evaluation_metric,
                        gold_label_dictionary=gold_label_dictionary_for_eval,
                    )
                    result_line += f"\t{test_eval_result.loss}\t{test_eval_result.log_line}"
                    log.info(
                        f"TEST : loss {test_eval_result.loss} -"
                        f" {main_evaluation_metric[1]}"
                        f" ({main_evaluation_metric[0]}) "
                        f" {round(test_eval_result.main_score, 4)}"
                    )

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.test, embeddings_storage_mode, dynamic_embeddings)

                    if use_tensorboard:
                        writer.add_scalar("test_loss", test_eval_result.loss, epoch)
                        writer.add_scalar("test_score", test_eval_result.main_score, epoch)
                        for (
                            metric_class_avg_type,
                            metric_type,
                        ) in metrics_for_tensorboard:
                            writer.add_scalar(
                                f"test_{metric_class_avg_type}_{metric_type}",
                                test_eval_result.classification_report[metric_class_avg_type][metric_type],
                                epoch,
                            )

                # determine if this is the best model or if we need to anneal
                current_epoch_has_best_model_so_far = False
                # default mode: anneal against dev score
                if not train_with_dev and not anneal_against_dev_loss:
                    if dev_score > best_validation_score:
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = dev_score

                    if isinstance(scheduler, AnnealOnPlateau):
                        scheduler.step(dev_score, dev_eval_result.loss)

                # alternative: anneal against dev loss
                if not train_with_dev and anneal_against_dev_loss:
                    if dev_eval_result.loss < best_validation_score:
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = dev_eval_result.loss

                    if isinstance(scheduler, AnnealOnPlateau):
                        scheduler.step(dev_eval_result.loss)

                # alternative: anneal against train loss
                if train_with_dev:
                    if train_loss < best_validation_score:
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = train_loss

                    if isinstance(scheduler, AnnealOnPlateau):
                        scheduler.step(train_loss)

                train_loss_history.append(train_loss)

                # determine bad epoch number
                try:
                    bad_epochs = scheduler.num_bad_epochs
                except AttributeError:
                    bad_epochs = 0

                new_learning_rate = [group["lr"] for group in optimizer.param_groups]

                if any([new_lr != prev_lr for new_lr, prev_lr in zip(new_learning_rate, previous_learning_rate)]):
                    bad_epochs = patience + 1

                    # lr unchanged
                    if all(
                        [
                            prev_lr == initial_lr
                            for prev_lr, initial_lr in zip(previous_learning_rate, initial_learning_rate)
                        ]
                    ):
                        bad_epochs += initial_extra_patience

                # log bad epochs
                log.info(f"BAD EPOCHS (no improvement): {bad_epochs}")

                if loss_txt is not None:
                    # output log file
                    with open(loss_txt, "a") as f:

                        # make headers on first epoch
                        if epoch == 1:
                            f.write("EPOCH\tTIMESTAMP\tBAD_EPOCHS" "\tLEARNING_RATE\tTRAIN_LOSS")

                            if log_train:
                                f.write("\tTRAIN_" + "\tTRAIN_".join(train_eval_result.log_header.split("\t")))

                            if log_train_part:
                                f.write(
                                    "\tTRAIN_PART_LOSS\tTRAIN_PART_"
                                    + "\tTRAIN_PART_".join(train_part_eval_result.log_header.split("\t"))
                                )

                            if log_dev:
                                f.write("\tDEV_LOSS\tDEV_" + "\tDEV_".join(dev_eval_result.log_header.split("\t")))

                            if log_test:
                                f.write("\tTEST_LOSS\tTEST_" + "\tTEST_".join(test_eval_result.log_header.split("\t")))

                        lr_info = ",".join([f"{lr:.4f}" for lr in current_learning_rate])

                        f.write(
                            f"\n{epoch}\t{datetime.datetime.now():%H:%M:%S}"
                            f"\t{bad_epochs}"
                            f"\t{lr_info}\t{train_loss}"
                        )
                        f.write(result_line)

                # if checkpoint is enabled, save model at each epoch
                if checkpoint and not param_selection_mode:
                    self.model.save(base_path / "checkpoint.pt", checkpoint=True)

                # Check whether to save best model
                if (
                    (not train_with_dev or anneal_with_restarts or anneal_with_prestarts)
                    and not param_selection_mode
                    and current_epoch_has_best_model_so_far
                    and not use_final_model_for_eval
                ):
                    log.info("saving best model")
                    self.model.save(base_path / "best-model.pt", checkpoint=save_optimizer_state)

                    if anneal_with_prestarts:
                        current_state_dict = self.model.state_dict()
                        self.model.load_state_dict(last_epoch_model_state_dict)
                        self.model.save(base_path / "pre-best-model.pt")
                        self.model.load_state_dict(current_state_dict)

            if use_swa:
                import torchcontrib

                cast(torchcontrib.optim.SWA, optimizer).swap_swa_sgd()

            # if we do not use dev data for model selection, save final model
            if save_final_model and not param_selection_mode:
                self.model.save(base_path / "final-model.pt", checkpoint=save_optimizer_state)

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")

            if not param_selection_mode:
                log.info("Saving model ...")
                self.model.save(base_path / "final-model.pt", checkpoint=save_optimizer_state)
                log.info("Done.")
        except Exception:
            if create_file_logs:
                log_handler.close()
                log.removeHandler(log_handler)
            raise
        finally:
            if use_tensorboard:
                writer.close()

        # test best model if test data is present
        if self.corpus.test and not train_with_test:
            final_score = self.final_test(
                base_path=base_path,
                eval_mini_batch_size=eval_batch_size,
                num_workers=num_workers,
                main_evaluation_metric=main_evaluation_metric,
                gold_label_dictionary_for_eval=gold_label_dictionary_for_eval,
            )
        else:
            final_score = 0
            log.info("Test data not provided setting final score to 0")

        if create_file_logs:
            log_handler.close()
            log.removeHandler(log_handler)

        return {
            "test_score": final_score,
            "dev_score_history": dev_score_history,
            "train_loss_history": train_loss_history,
            "dev_loss_history": dev_loss_history,
        }

    def resume(
        self,
        model: Model,
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
        **trainer_args,
    ):

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
        )

        log.info(test_results.log_line)
        log.info(test_results.detailed_results)
        log_line(log)

        # if we are training over multiple datasets, do evaluation for each
        if isinstance(self.corpus, MultiCorpus):
            for subcorpus in self.corpus.corpora:
                log_line(log)
                if subcorpus.test:
                    subcorpus_results = self.model.evaluate(
                        subcorpus.test,
                        gold_label_type=self.model.label_type,
                        mini_batch_size=eval_mini_batch_size,
                        num_workers=num_workers,
                        out_path=base_path / f"{subcorpus.name}-test.tsv",
                        embedding_storage_mode="none",
                        main_evaluation_metric=main_evaluation_metric,
                    )
                    log.info(subcorpus.name)
                    log.info(subcorpus_results.log_line)

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
                loss = self.model.forward_loss(batch)
                if isinstance(loss, tuple):
                    loss = loss[0]

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
