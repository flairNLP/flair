import datetime
import os
import random
import logging
from typing import List

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from flair.data import Sentence, TaggedCorpus, Dictionary
from flair.models.text_classification_model import TextClassifier
from flair.training_utils import init_output_file, clear_embeddings, WeightExtractor, Metric

log = logging.getLogger(__name__)


class TextClassifierTrainer:
    """
    Training class to train and evaluate a text classification model.
    """

    def __init__(self, model: TextClassifier, corpus: TaggedCorpus, label_dict: Dictionary,
                 test_mode: bool = False) -> None:
        self.model: TextClassifier = model
        self.corpus: TaggedCorpus = corpus
        self.label_dict: Dictionary = label_dict
        self.test_mode: bool = test_mode

    def train(self,
              base_path: str,
              learning_rate: float = 0.1,
              mini_batch_size: int = 32,
              max_epochs: int = 50,
              anneal_factor: float = 0.5,
              patience: int = 5,
              train_with_dev: bool = False,
              embeddings_in_memory: bool = False,
              checkpoint: bool = False,
              save_final_model: bool = True,
              anneal_with_restarts: bool = False,
              eval_on_train: bool = True):
        """
        Trains a text classification model using the training data of the corpus.
        :param base_path: the directory to which any results should be written to
        :param learning_rate: the learning rate
        :param mini_batch_size: the mini batch size
        :param max_epochs: the maximum number of epochs to train
        :param anneal_factor: learning rate will be decreased by this factor
        :param patience: number of 'bad' epochs before learning rate gets decreased
        :param train_with_dev: boolean indicating, if the dev data set should be used for training or not
        :param embeddings_in_memory: boolean indicating, if embeddings should be kept in memory or not
        :param checkpoint: boolean indicating, whether the model should be save after every epoch or not
        :param save_final_model: boolean indicating, whether the final model should be saved or not
        :param anneal_with_restarts: boolean indicating, whether the best model should be reloaded once the learning
        rate changed or not
        :param eval_on_train: boolean value indicating, if evaluation metrics should be calculated on training data set
        or not
        """

        loss_txt = init_output_file(base_path, 'loss.tsv')
        with open(loss_txt, 'a') as f:
            f.write('EPOCH\tTIMESTAMP\tTRAIN_LOSS\t{}\tDEV_LOSS\t{}\tTEST_LOSS\t{}\n'.format(
                Metric.tsv_header('TRAIN'), Metric.tsv_header('DEV'), Metric.tsv_header('TEST')))

        weight_extractor = WeightExtractor(base_path)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        anneal_mode = 'min' if train_with_dev else 'max'
        scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer, factor=anneal_factor, patience=patience,
                                                         mode=anneal_mode)

        train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data.extend(self.corpus.dev)

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            previous_learning_rate = learning_rate

            for epoch in range(max_epochs):
                log.info('-' * 100)

                bad_epochs = scheduler.num_bad_epochs
                for group in optimizer.param_groups:
                    learning_rate = group['lr']

                # reload last best model if annealing with restarts is enabled
                if learning_rate != previous_learning_rate and anneal_with_restarts and \
                        os.path.exists(base_path + "/best-model.pt"):
                    log.info('Resetting to best model ...')
                    self.model.load_from_file(base_path + "/best-model.pt")

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if learning_rate < 0.001:
                    log.info('Learning rate too small - quitting training!')
                    break

                if not self.test_mode:
                    random.shuffle(train_data)

                self.model.train()

                batches = [self.corpus.train[x:x + mini_batch_size] for x in
                           range(0, len(self.corpus.train), mini_batch_size)]

                current_loss: float = 0
                seen_sentences = 0
                modulo = max(1, int(len(batches) / 10))

                for batch_no, batch in enumerate(batches):
                    loss = self.model.forward_and_loss(batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    seen_sentences += len(batch)
                    current_loss += loss.item()

                    clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

                    if batch_no % modulo == 0:
                        log.info("epoch {0} - iter {1}/{2} - loss {3:.8f}".format(
                            epoch + 1, batch_no, len(batches), current_loss / seen_sentences))
                        iteration = epoch * len(batches) + batch_no
                        weight_extractor.extract_weights(self.model.state_dict(), iteration)

                current_loss /= len(train_data)

                self.model.eval()

                # if checkpoint is enable, save model at each epoch
                if checkpoint:
                    self.model.save(base_path + "/checkpoint.pt")

                log.info('-' * 100)
                log.info(
                    "EPOCH {0}: lr {1:.4f} - bad epochs {2}".format(epoch + 1, learning_rate, bad_epochs))

                dev_metric = train_metric = None
                dev_loss = '_'
                train_loss = current_loss

                if eval_on_train:
                    train_metric, train_loss = self._calculate_evaluation_results_for(
                        'TRAIN', self.corpus.train, embeddings_in_memory, mini_batch_size)

                if not train_with_dev:
                    dev_metric, dev_loss = self._calculate_evaluation_results_for(
                        'DEV', self.corpus.dev, embeddings_in_memory, mini_batch_size)

                self._calculate_evaluation_results_for('TEST', self.corpus.test, embeddings_in_memory, mini_batch_size)

                with open(loss_txt, 'a') as f:
                    train_metric_str = train_metric.to_tsv() if train_metric is not None else Metric.to_empty_tsv()
                    dev_metric_str = dev_metric.to_tsv() if dev_metric is not None else Metric.to_empty_tsv()
                    f.write('{}\t{:%H:%M:%S}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        epoch, datetime.datetime.now(), train_loss, train_metric_str, dev_loss, dev_metric_str, '_',
                        Metric.to_empty_tsv()))

                # anneal against train loss if training with dev, otherwise anneal against dev score
                scheduler.step(current_loss) if train_with_dev else scheduler.step(dev_metric.f_score())

                current_score = dev_metric.f_score() if not train_with_dev else train_metric.f_score()

                # if we use dev data, remember best model based on dev evaluation score
                if not train_with_dev and current_score == scheduler.best:
                    self.model.save(base_path + "/best-model.pt")

            if save_final_model:
                self.model.save(base_path + "/final-model.pt")

            log.info('-' * 100)
            log.info('Testing using best model ...')

            self.model.eval()

            if os.path.exists(base_path + "/best-model.pt"):
                self.model = TextClassifier.load_from_file(base_path + "/best-model.pt")

            test_metric, test_loss = self.evaluate(
                self.corpus.test, mini_batch_size=mini_batch_size, eval_class_metrics=True,
                embeddings_in_memory=embeddings_in_memory, metric_name='TEST')

            test_metric.print()
            self.model.train()

            log.info('-' * 100)

        except KeyboardInterrupt:
            log.info('-' * 100)
            log.info('Exiting from training early.')
            log.info('Saving model ...')
            with open(base_path + "/final-model.pt", 'wb') as model_save_file:
                torch.save(self.model, model_save_file, pickle_protocol=4)
                model_save_file.close()
            log.info('Done.')

    def _calculate_evaluation_results_for(self, dataset_name, dataset, embeddings_in_memory, mini_batch_size):
        metric, loss = self.evaluate(dataset, mini_batch_size=mini_batch_size,
                                     embeddings_in_memory=embeddings_in_memory, metric_name=dataset_name)

        f_score = metric.f_score()
        acc = metric.accuracy()

        log.info("{0:<5}: loss {1:.8f} - f-score {2:.4f} - acc {3:.4f}".format(
            dataset_name, loss, f_score, acc))

        return metric, loss

    def evaluate(self, sentences: List[Sentence], eval_class_metrics: bool = False, mini_batch_size: int = 32,
                 embeddings_in_memory: bool = False, metric_name: str = 'MICRO_AVG') -> (dict, float):
        """
        Evaluates the model with the given list of sentences.
        :param sentences: the list of sentences
        :param eval_class_metrics: boolean indicating whether to print class metrics or not
        :param mini_batch_size: the mini batch size to use
        :param embeddings_in_memory: boolean value indicating, if embeddings should be kept in memory or not
        :param metric_name: the name of the metrics to compute
        :return: list of metrics, and the loss
        """
        with torch.no_grad():
            eval_loss = 0

            batches = [sentences[x:x + mini_batch_size] for x in
                       range(0, len(sentences), mini_batch_size)]

            metric = Metric(metric_name)

            for batch in batches:
                labels, loss = self.model.predict_eval(sentences)

                clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

                eval_loss += loss

                for predictions, true_values in zip([[label.value for label in sent_labels] for sent_labels in labels],
                                                    [sentence.get_label_names() for sentence in batch]):
                    for prediction in predictions:
                        if prediction in true_values:
                            metric.tp()
                            if eval_class_metrics: metric.tp(prediction)
                        else:
                            metric.fp()
                            if eval_class_metrics: metric.fp(prediction)

                    for true_value in true_values:
                        if true_value not in predictions:
                            metric.fn()
                            if eval_class_metrics: metric.fn(true_value)
                        else:
                            metric.tn()
                            if eval_class_metrics: metric.tn(true_value)

            eval_loss /= len(sentences)

            return metric, eval_loss
