from pathlib import Path
from ray.tune import Trainable
from ray.tune.trial import Trial

from flair.datasets import DataLoader
from flair.trainers import ModelTrainer

from flair.training_utils import Result

import logging

logger = logging.getLogger("flair")


class FlairTune(Trainable):
    def _train_iteration(self):
        if not "epoch" in self.__dict__:
            self.epoch = 0
        # run training code
        loss = self.trainer._train_one_epoch(
            self.epoch, embeddings_storage_mode=self.embeddings_storage_mode
        )
        self.epoch += 1
        return loss

    def _test(self):
        if not "test" in self.__dict__:
            self.test = DataLoader(self.corpus.dev)

        self.trainer.model.eval()
        results, test_loss = self.trainer.model.evaluate(
            self.test,
            embeddings_storage_mode=self.embeddings_storage_mode,
            out_path=Path(self._logdir) / "test.txt",
        )

        results: Result = results

        # determine learning rate annealing through scheduler
        if results.main_score != 0.0:
            self.trainer.scheduler.step(results.main_score)
        for group in self.trainer.optimizer.param_groups:
            learning_rate = group["lr"]

        with open(Path(self._logdir) / "eval.txt", "a", encoding="utf-8") as outfile:
            if self.epoch == 1:
                outfile.write(results.log_header + "\n")
            outfile.write(results.log_line + "\n")

        result_dict = {"mean_accuracy": results.main_score, "1-lr": 1 - learning_rate}
        return result_dict

    def _train(self):
        if not "embeddings_storage_mode" in self.__dict__:
            self.embeddings_storage_mode = "cpu"
        loss = self._train_iteration()
        result_dict = self._test()
        result_dict["mean_loss"] = loss
        return result_dict

    def _save(self, checkpoint_dir):
        path = Path(checkpoint_dir) / "checkpoint.pt"
        self.trainer.save_checkpoint(path)
        return str(path)

    def _restore(self, checkpoint_prefix):
        self.trainer = ModelTrainer.load_checkpoint(
            Path(checkpoint_prefix), self.corpus
        )


def write_experiment_summary(trials, result_file_path):

    with open(result_file_path, "w", encoding="utf-8") as outfile:
        header = None
        for trial in trials:
            trial: Trial = trial
            if header is None:
                header = "\t".join(sorted(trial.config.keys()))
                header += "\tfinal LR\tmean_loss\titerations\tmean_accuracy\n"
                outfile.write(header)
                logger.info(header)

            result_line = ""
            for config_key in sorted(trial.config.keys()):
                if type(trial.config[config_key]) == float:
                    result_line += str(round(trial.config[config_key], 2)) + "\t"

                else:
                    result_line += str(trial.config[config_key]) + "\t"

            result_line += f"{1 - trial.last_result['1-lr']}\t{trial.last_result['mean_loss']}\t{trial.last_result['training_iteration']}\t{trial.last_result['mean_accuracy']}\n"
            outfile.write(result_line)
            logger.info(result_line)
        outfile.close()
