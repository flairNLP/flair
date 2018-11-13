from typing import List, Union

import datetime
import os
import random
import logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import flair
import flair.nn
from flair.data import Sentence, Token, TaggedCorpus, Label
from flair.models import TextClassifier, SequenceTagger
from flair.training_utils import Metric, init_output_file, WeightExtractor, clear_embeddings

log = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(self, model: flair.nn.Model, corpus: TaggedCorpus) -> None:
        self.model: flair.nn.Model = model
        self.corpus: TaggedCorpus = corpus

    def train(self,
              base_path: str,
              learning_rate: float = 0.1,
              mini_batch_size: int = 32,
              max_epochs: int = 100,
              anneal_factor: float = 0.5,
              patience: int = 4,
              train_with_dev: bool = False,
              embeddings_in_memory: bool = True,
              checkpoint: bool = False,
              save_final_model: bool = True,
              anneal_with_restarts: bool = False,
              test_mode: bool = False,
              ):

        self._log_line()
        log.info(f'Evaluation method: {self.model.evaluation_metric().name}')

        loss_txt = init_output_file(base_path, 'loss.tsv')
        with open(loss_txt, 'a') as f:
            f.write(f'EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS\t{Metric.tsv_header("TRAIN")}\tDEV_LOSS\t{Metric.tsv_header("DEV")}'
                    f'\tTEST_LOSS\t{Metric.tsv_header("TEST")}\n')

        weight_extractor = WeightExtractor(base_path)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # annealing scheduler
        anneal_mode = 'min' if train_with_dev else 'max'
        scheduler = ReduceLROnPlateau(optimizer, factor=anneal_factor, patience=patience, mode=anneal_mode,
                                      verbose=True)

        train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data.extend(self.corpus.dev)

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            previous_learning_rate = learning_rate

            for epoch in range(0, max_epochs):
                self._log_line()

                bad_epochs = scheduler.num_bad_epochs
                for group in optimizer.param_groups:
                    learning_rate = group['lr']

                # reload last best model if annealing with restarts is enabled
                if learning_rate != previous_learning_rate and anneal_with_restarts and \
                        os.path.exists(base_path + '/best-model.pt'):
                    log.info('resetting to best model')
                    self.model.load_from_file(base_path + '/best-model.pt')

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if learning_rate < 0.001:
                    self._log_line()
                    log.info('learning rate too small - quitting training!')
                    self._log_line()
                    break

                if not test_mode:
                    random.shuffle(train_data)

                batches = [train_data[x:x + mini_batch_size] for x in range(0, len(train_data), mini_batch_size)]

                self.model.train()

                current_loss: float = 0
                seen_sentences = 0
                modulo = max(1, int(len(batches) / 10))

                for batch_no, batch in enumerate(batches):
                    loss = self.model.forward_loss(batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    seen_sentences += len(batch)
                    current_loss += loss.item()

                    clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

                    if batch_no % modulo == 0:
                        log.info(f'epoch {epoch + 1} - iter {batch_no}/{len(batches)} - loss '
                                 f'{current_loss / seen_sentences:.8f}')
                        iteration = epoch * len(batches) + batch_no
                        weight_extractor.extract_weights(self.model.state_dict(), iteration)

                current_loss /= len(train_data)

                self.model.eval()

                # if checkpoint is enable, save model at each epoch
                if checkpoint:
                    self.model.save(base_path + '/checkpoint.pt')

                self._log_line()
                log.info(f'EPOCH {epoch + 1}: lr {learning_rate:.4f} - bad epochs {bad_epochs}')

                dev_metric = None
                dev_loss = '_'

                train_metric, train_loss = self._calculate_evaluation_results_for(
                    'TRAIN', self.corpus.train, embeddings_in_memory, mini_batch_size)

                test_metric, test_loss = self._calculate_evaluation_results_for(
                    'TEST', self.corpus.test, embeddings_in_memory, mini_batch_size)

                if not train_with_dev:
                    dev_metric, dev_loss = self._calculate_evaluation_results_for(
                        'DEV', self.corpus.dev, embeddings_in_memory, mini_batch_size)

                with open(loss_txt, 'a') as f:
                    train_metric_str = train_metric.to_tsv() if train_metric is not None else Metric.to_empty_tsv()
                    dev_metric_str = dev_metric.to_tsv() if dev_metric is not None else Metric.to_empty_tsv()
                    test_metric_str = test_metric.to_tsv() if test_metric is not None else Metric.to_empty_tsv()
                    f.write(f'{epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t{train_loss}\t{train_metric_str}\t{dev_loss}'
                            f'\t{dev_metric_str}\t_\t{test_metric_str}\n')

                if train_with_dev:
                    current_score = train_metric.micro_avg_f_score() \
                        if self.model.evaluation_metric() == flair.nn.EvaluationMetric.ACCURACY \
                        else train_metric.micro_avg_accuracy()
                else:
                    current_score = dev_metric.micro_avg_f_score() \
                        if self.model.evaluation_metric() == flair.nn.EvaluationMetric.ACCURACY \
                        else dev_metric.micro_avg_accuracy()

                # anneal against train loss if training with dev, otherwise anneal against dev score
                scheduler.step(current_loss) if train_with_dev else scheduler.step(current_score)

                # if we use dev data, remember best model based on dev evaluation score
                if not train_with_dev and current_score == scheduler.best:
                    self.model.save(base_path + '/best-model.pt')

            # if we do not use dev data for model selection, save final model
            if save_final_model:
                self.model.save(base_path + '/final-model.pt')

        except KeyboardInterrupt:
            self._log_line()
            log.info('Exiting from training early.')
            log.info('Saving model ...')
            self.model.save(base_path + "/final-model.pt")
            log.info('Done.')
            self._log_line()

        log.info('Testing using best model ...')

        self.model.eval()

        if os.path.exists(base_path + "/best-model.pt"):
            if isinstance(self.model, TextClassifier):
                self.model = TextClassifier.load_from_file(base_path + "/best-model.pt")
            if isinstance(self.model, SequenceTagger):
                self.model = SequenceTagger.load_from_file(base_path + "/best-model.pt")

        test_metric, test_loss = self.evaluate(self.model, self.corpus.test, mini_batch_size=mini_batch_size,
                                               embeddings_in_memory=embeddings_in_memory)

        log.info(f'MICRO_AVG: acc {test_metric.micro_avg_accuracy()} - f1-score {test_metric.micro_avg_f_score()}')
        log.info(f'MARCO_AVG: acc {test_metric.macro_avg_accuracy()} - f1-score {test_metric.macro_avg_f_score()}')
        for class_name in test_metric.get_classes():
            log.info(f'{class_name:<10} tp: {test_metric.get_tp(class_name)} - fp: {test_metric.get_fp(class_name)} - '
                     f'fn: {test_metric.get_fn(class_name)} - tn: {test_metric.get_tn(class_name)} - precision: '
                     f'{test_metric.precision(class_name):.4f} - recall: {test_metric.recall(class_name):.4f} - '
                     f'accuracy: {test_metric.accuracy(class_name):.4f} - f1-score: '
                     f'{test_metric.f_score(class_name):.4f}')
        self._log_line()

    def _calculate_evaluation_results_for(self, dataset_name, dataset, embeddings_in_memory, mini_batch_size):
        metric, loss = ModelTrainer.evaluate(self.model, dataset, mini_batch_size=mini_batch_size,
                                     embeddings_in_memory=embeddings_in_memory)

        f_score = metric.micro_avg_f_score()
        acc = metric.micro_avg_accuracy()

        log.info(f'{dataset_name:<5}: loss {loss:.8f} - f-score {f_score:.4f} - acc {acc:.4f}')

        return metric, loss

    @staticmethod
    def evaluate(model: flair.nn.Model, data_set: List[Sentence], mini_batch_size=32, embeddings_in_memory=True) -> (dict, float):
        if isinstance(model, TextClassifier):
            return ModelTrainer._evaluate_text_classifier(model, data_set, mini_batch_size, embeddings_in_memory)
        elif isinstance(model, SequenceTagger):
            return ModelTrainer._evaluate_sequence_tagger(model, data_set, mini_batch_size, embeddings_in_memory)


    @staticmethod
    def _evaluate_sequence_tagger(model, sentences: List[Sentence], eval_batch_size: int=32, embeddings_in_memory: bool=True) -> (dict, float):

        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0
            batches = [sentences[x:x + eval_batch_size] for x in range(0, len(sentences), eval_batch_size)]

            metric = Metric('Evaluation')

            for batch in batches:
                batch_no += 1

                tags, loss = model.forward_labels_and_loss(batch)

                eval_loss += loss

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag_label('predicted', tag)

                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans(model.tag_type)]
                    # make list of predicted tags
                    predicted_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans('predicted')]

                    # check for true positives, false positives and false negatives
                    for tag, prediction in predicted_tags:
                        if (tag, prediction) in gold_tags:
                            metric.tp(tag)
                        else:
                            metric.fp(tag)

                    for tag, gold in gold_tags:
                        if (tag, gold) not in predicted_tags:
                            metric.fn(tag)
                        else:
                            metric.tn(tag)

                clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

            eval_loss /= len(sentences)

            return metric, eval_loss

    @staticmethod
    def _evaluate_text_classifier(model: flair.nn.Model, sentences: List[Sentence], mini_batch_size: int = 32,
                 embeddings_in_memory: bool = False) -> (dict, float):

        with torch.no_grad():
            eval_loss = 0

            batches = [sentences[x:x + mini_batch_size] for x in
                       range(0, len(sentences), mini_batch_size)]

            metric = Metric('Evaluation')

            for batch in batches:
                labels, loss = model.forward_labels_and_loss(sentences)

                clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

                eval_loss += loss

                for predictions, true_values in zip([[label.value for label in sent_labels] for sent_labels in labels],
                                                    [sentence.get_label_names() for sentence in batch]):
                    for prediction in predictions:
                        if prediction in true_values:
                            metric.tp(prediction)
                        else:
                            metric.fp(prediction)

                    for true_value in true_values:
                        if true_value not in predictions:
                            metric.fn(true_value)
                        else:
                            metric.tn(true_value)

            eval_loss /= len(sentences)

            return metric, eval_loss

    def _log_line(self):
        log.info('-' * 100)
