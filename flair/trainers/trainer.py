from pathlib import Path
from typing import List, Union

import datetime
import random
import logging
from torch.optim.sgd import SGD

import flair
import flair.nn
from flair.data import Sentence, Token, MultiCorpus, Corpus
from flair.models import TextClassifier, SequenceTagger
from flair.training_utils import Metric, init_output_file, WeightExtractor, clear_embeddings, EvaluationMetric, \
    log_line, add_file_handler
from flair.optim import *


log = logging.getLogger('flair')


class ModelTrainer:

    def __init__(self,
                 model: flair.nn.Model,
                 corpus: Corpus,
                 optimizer: Optimizer = SGD,
                 epoch:int = 0,
                 loss: float = 10000.0,
                 optimizer_state: dict = None,
                 scheduler_state: dict = None
                 ):
        self.model: flair.nn.Model = model
        self.corpus: Corpus = corpus
        self.optimizer: Optimizer = optimizer
        self.epoch: int = epoch
        self.loss: float = loss
        self.scheduler_state: dict = scheduler_state
        self.optimizer_state: dict = optimizer_state

    def train(self,
              base_path: Union[Path, str],
              evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
              learning_rate: float = 0.1,
              mini_batch_size: int = 32,
              eval_mini_batch_size: int = None,
              max_epochs: int = 100,
              anneal_factor: float = 0.5,
              patience: int = 3,
              anneal_against_train_loss: bool = True,
              train_with_dev: bool = False,
              monitor_train: bool = False,
              embeddings_in_memory: bool = True,
              checkpoint: bool = False,
              save_final_model: bool = True,
              anneal_with_restarts: bool = False,
              test_mode: bool = False,
              param_selection_mode: bool = False,
              **kwargs
              ) -> dict:

        if eval_mini_batch_size is None:
            eval_mini_batch_size = mini_batch_size

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        add_file_handler(log, base_path / 'training.log')

        log_line(log)
        log.info(f'Evaluation method: {evaluation_metric.name}')

        if not param_selection_mode:
            loss_txt = init_output_file(base_path, 'loss.tsv')
            with open(loss_txt, 'a') as f:
                f.write(
                    f'EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS\t{Metric.tsv_header("TRAIN")}\tDEV_LOSS\t{Metric.tsv_header("DEV")}'
                    f'\tTEST_LOSS\t{Metric.tsv_header("TEST")}\n')

            weight_extractor = WeightExtractor(base_path)

        optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, **kwargs)
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)

        # annealing scheduler
        anneal_mode = 'min' if anneal_against_train_loss else 'max'
        if isinstance(optimizer, (AdamW, SGDW)):
            scheduler = ReduceLRWDOnPlateau(optimizer, factor=anneal_factor,
                                            patience=patience, mode=anneal_mode,
                                            verbose=True)
        else:
            scheduler = ReduceLROnPlateau(optimizer, factor=anneal_factor,
                                          patience=patience, mode=anneal_mode,
                                          verbose=True)
        if self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)

        train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data.extend(self.corpus.dev)

        dev_score_history = []
        dev_loss_history = []
        train_loss_history = []

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            previous_learning_rate = learning_rate

            for epoch in range(0 + self.epoch, max_epochs + self.epoch):
                log_line(log)

                try:
                    bad_epochs = scheduler.num_bad_epochs
                except:
                    bad_epochs = 0
                for group in optimizer.param_groups:
                    learning_rate = group['lr']

                # reload last best model if annealing with restarts is enabled
                if learning_rate != previous_learning_rate and anneal_with_restarts and \
                        (base_path / 'best-model.pt').exists():
                    log.info('resetting to best model')
                    self.model.load_from_file(base_path / 'best-model.pt')

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if learning_rate < 0.0001:
                    log_line(log)
                    log.info('learning rate too small - quitting training!')
                    log_line(log)
                    break

                if not test_mode:
                    random.shuffle(train_data)

                batches = [train_data[x:x + mini_batch_size] for x in range(0, len(train_data), mini_batch_size)]

                self.model.train()

                train_loss: float = 0
                seen_sentences = 0
                modulo = max(1, int(len(batches) / 10))

                for batch_no, batch in enumerate(batches):
                    loss = self.model.forward_loss(batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    seen_sentences += len(batch)
                    train_loss += loss.item()

                    clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

                    if batch_no % modulo == 0:
                        log.info(f'epoch {epoch + 1} - iter {batch_no}/{len(batches)} - loss '
                                 f'{train_loss / seen_sentences:.8f}')
                        iteration = epoch * len(batches) + batch_no
                        if not param_selection_mode:
                            weight_extractor.extract_weights(self.model.state_dict(), iteration)

                train_loss /= len(train_data)

                self.model.eval()

                log_line(log)
                log.info(f'EPOCH {epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate:.4f} - bad epochs {bad_epochs}')

                dev_metric = None
                dev_loss = '_'

                train_metric = None
                if monitor_train:
                    train_metric, train_loss = self._calculate_evaluation_results_for(
                        'TRAIN', self.corpus.train, evaluation_metric, embeddings_in_memory, eval_mini_batch_size)

                if not train_with_dev:
                    dev_metric, dev_loss = self._calculate_evaluation_results_for(
                        'DEV', self.corpus.dev, evaluation_metric, embeddings_in_memory, eval_mini_batch_size)

                if not param_selection_mode:
                    test_metric, test_loss = self._calculate_evaluation_results_for(
                        'TEST', self.corpus.test, evaluation_metric, embeddings_in_memory, eval_mini_batch_size,
                        base_path / 'test.tsv')

                if not param_selection_mode:
                    with open(loss_txt, 'a') as f:
                        train_metric_str = train_metric.to_tsv() if train_metric is not None else Metric.to_empty_tsv()
                        dev_metric_str = dev_metric.to_tsv() if dev_metric is not None else Metric.to_empty_tsv()
                        test_metric_str = test_metric.to_tsv() if test_metric is not None else Metric.to_empty_tsv()
                        f.write(
                            f'{epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t'
                            f'{train_loss}\t{train_metric_str}\t{dev_loss}\t{dev_metric_str}\t_\t{test_metric_str}\n')

                # calculate scores using dev data if available
                dev_score = 0.
                if not train_with_dev:
                    if evaluation_metric == EvaluationMetric.MACRO_ACCURACY:
                        dev_score = dev_metric.macro_avg_accuracy()
                    elif evaluation_metric == EvaluationMetric.MICRO_ACCURACY:
                        dev_score = dev_metric.micro_avg_accuracy()
                    elif evaluation_metric == EvaluationMetric.MACRO_F1_SCORE:
                        dev_score = dev_metric.macro_avg_f_score()
                    else:
                        dev_score = dev_metric.micro_avg_f_score()

                    # append dev score to score history
                    dev_score_history.append(dev_score)
                    dev_loss_history.append(dev_loss.item())

                # anneal against train loss if training with dev, otherwise anneal against dev score
                current_score = train_loss if anneal_against_train_loss else dev_score

                scheduler.step(current_score)

                train_loss_history.append(train_loss)

                # if checkpoint is enable, save model at each epoch
                if checkpoint and not param_selection_mode:
                    self.model.save_checkpoint(base_path / 'checkpoint.pt',
                                               optimizer.state_dict(), scheduler.state_dict(),
                                               epoch + 1, train_loss)

                # if we use dev data, remember best model based on dev evaluation score
                if not train_with_dev and not param_selection_mode and current_score == scheduler.best:
                    self.model.save(base_path / 'best-model.pt')

            # if we do not use dev data for model selection, save final model
            if save_final_model and not param_selection_mode :
                self.model.save(base_path / 'final-model.pt')

        except KeyboardInterrupt:
            log_line(log)
            log.info('Exiting from training early.')
            if not param_selection_mode:
                log.info('Saving model ...')
                self.model.save(base_path / 'final-model.pt')
                log.info('Done.')

        # test best model on test data
        final_score = self.final_test(base_path, embeddings_in_memory, evaluation_metric, eval_mini_batch_size)

        return {'test_score': final_score,
                'dev_score_history': dev_score_history,
                'train_loss_history': train_loss_history,
                'dev_loss_history': dev_loss_history}

    def final_test(self,
                   base_path: Path,
                   embeddings_in_memory: bool,
                   evaluation_metric: EvaluationMetric,
                   eval_mini_batch_size: int):

        log_line(log)
        log.info('Testing using best model ...')

        self.model.eval()

        if (base_path / 'best-model.pt').exists():
            if isinstance(self.model, TextClassifier):
                self.model = TextClassifier.load_from_file(base_path / 'best-model.pt')
            if isinstance(self.model, SequenceTagger):
                self.model = SequenceTagger.load_from_file(base_path / 'best-model.pt')

        test_metric, test_loss = self.evaluate(self.model, self.corpus.test, eval_mini_batch_size=eval_mini_batch_size,
                                               embeddings_in_memory=embeddings_in_memory)

        log.info(f'MICRO_AVG: acc {test_metric.micro_avg_accuracy()} - f1-score {test_metric.micro_avg_f_score()}')
        log.info(f'MACRO_AVG: acc {test_metric.macro_avg_accuracy()} - f1-score {test_metric.macro_avg_f_score()}')
        for class_name in test_metric.get_classes():
            log.info(f'{class_name:<10} tp: {test_metric.get_tp(class_name)} - fp: {test_metric.get_fp(class_name)} - '
                     f'fn: {test_metric.get_fn(class_name)} - tn: {test_metric.get_tn(class_name)} - precision: '
                     f'{test_metric.precision(class_name):.4f} - recall: {test_metric.recall(class_name):.4f} - '
                     f'accuracy: {test_metric.accuracy(class_name):.4f} - f1-score: '
                     f'{test_metric.f_score(class_name):.4f}')
        log_line(log)

        # if we are training over multiple datasets, do evaluation for each
        if type(self.corpus) is MultiCorpus:
            for subcorpus in self.corpus.corpora:
                log_line(log)
                self._calculate_evaluation_results_for(subcorpus.name,
                                                       subcorpus.test,
                                                       evaluation_metric,
                                                       embeddings_in_memory,
                                                       eval_mini_batch_size,
                                                       base_path / 'test.tsv')

        # get and return the final test score of best model
        if evaluation_metric == EvaluationMetric.MACRO_ACCURACY:
            final_score = test_metric.macro_avg_accuracy()
        elif evaluation_metric == EvaluationMetric.MICRO_ACCURACY:
            final_score = test_metric.micro_avg_accuracy()
        elif evaluation_metric == EvaluationMetric.MACRO_F1_SCORE:
            final_score = test_metric.macro_avg_f_score()
        else:
            final_score = test_metric.micro_avg_f_score()

        return final_score

    def _calculate_evaluation_results_for(self,
                                          dataset_name: str,
                                          dataset: List[Sentence],
                                          evaluation_metric: EvaluationMetric,
                                          embeddings_in_memory: bool,
                                          eval_mini_batch_size: int,
                                          out_path: Path = None):

        metric, loss = ModelTrainer.evaluate(self.model, dataset, eval_mini_batch_size=eval_mini_batch_size,
                                             embeddings_in_memory=embeddings_in_memory, out_path=out_path)

        if evaluation_metric == EvaluationMetric.MACRO_ACCURACY or evaluation_metric == EvaluationMetric.MACRO_F1_SCORE:
            f_score = metric.macro_avg_f_score()
            acc = metric.macro_avg_accuracy()
        else:
            f_score = metric.micro_avg_f_score()
            acc = metric.micro_avg_accuracy()

        log.info(f'{dataset_name:<5}: loss {loss:.8f} - f-score {f_score:.4f} - acc {acc:.4f}')

        return metric, loss

    @staticmethod
    def evaluate(model: flair.nn.Model, data_set: List[Sentence],
                 eval_mini_batch_size: int = 32,
                 embeddings_in_memory: bool = True,
                 out_path: Path = None) -> (
            dict, float):
        if isinstance(model, TextClassifier):
            return ModelTrainer._evaluate_text_classifier(model, data_set, eval_mini_batch_size, embeddings_in_memory)
        elif isinstance(model, SequenceTagger):
            return ModelTrainer._evaluate_sequence_tagger(model, data_set, eval_mini_batch_size, embeddings_in_memory,
                                                          out_path)

    @staticmethod
    def _evaluate_sequence_tagger(model,
                                  sentences: List[Sentence],
                                  eval_mini_batch_size: int = 32,
                                  embeddings_in_memory: bool = True,
                                  out_path: Path = None) -> (dict, float):

        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0
            batches = [sentences[x:x + eval_mini_batch_size] for x in range(0, len(sentences), eval_mini_batch_size)]

            metric = Metric('Evaluation')

            lines: List[str] = []
            for batch in batches:
                batch_no += 1

                tags, loss = model.forward_labels_and_loss(batch)

                eval_loss += loss

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag_label('predicted', tag)

                        # append both to file for evaluation
                        eval_line = '{} {} {}\n'.format(token.text,
                                                        token.get_tag(model.tag_type).value, tag.value)
                        lines.append(eval_line)
                    lines.append('\n')
                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans(model.tag_type)]
                    # make list of predicted tags
                    predicted_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans('predicted')]

                    # check for true positives, false positives and false negatives
                    for tag, prediction in predicted_tags:
                        if (tag, prediction) in gold_tags:
                            metric.add_tp(tag)
                        else:
                            metric.add_fp(tag)

                    for tag, gold in gold_tags:
                        if (tag, gold) not in predicted_tags:
                            metric.add_fn(tag)
                        else:
                            metric.add_tn(tag)

                clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

            eval_loss /= len(sentences)

            if out_path is not None:
                with open(out_path, "w", encoding='utf-8') as outfile:
                    outfile.write(''.join(lines))

            return metric, eval_loss

    @staticmethod
    def _evaluate_text_classifier(model: flair.nn.Model,
                                  sentences: List[Sentence],
                                  eval_mini_batch_size: int = 32,
                                  embeddings_in_memory: bool = False) -> (dict, float):

        with torch.no_grad():
            eval_loss = 0

            batches = [sentences[x:x + eval_mini_batch_size] for x in
                       range(0, len(sentences), eval_mini_batch_size)]

            metric = Metric('Evaluation')

            for batch in batches:

                labels, loss = model.forward_labels_and_loss(batch)

                clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

                eval_loss += loss

                for predictions, true_values in zip([[label.value for label in sent_labels] for sent_labels in labels],
                                                    [sentence.get_label_names() for sentence in batch]):
                    for prediction in predictions:
                        if prediction in true_values:
                            metric.add_tp(prediction)
                        else:
                            metric.add_fp(prediction)

                    for true_value in true_values:
                        if true_value not in predictions:
                            metric.add_fn(true_value)
                        else:
                            metric.add_tn(true_value)

            eval_loss /= len(sentences)

            return metric, eval_loss

    @staticmethod
    def load_from_checkpoint(checkpoint_file: Path, model_type: str, corpus: Corpus, optimizer: Optimizer = SGD):
        if model_type == 'SequenceTagger':
            checkpoint = SequenceTagger.load_checkpoint(checkpoint_file)
            return ModelTrainer(checkpoint['model'], corpus, optimizer, epoch=checkpoint['epoch'],
                                loss=checkpoint['loss'], optimizer_state=checkpoint['optimizer_state_dict'],
                                scheduler_state=checkpoint['scheduler_state_dict'])

        if model_type == 'TextClassifier':
            checkpoint = TextClassifier.load_checkpoint(checkpoint_file)
            return ModelTrainer(checkpoint['model'], corpus, optimizer, epoch=checkpoint['epoch'],
                                loss=checkpoint['loss'], optimizer_state=checkpoint['optimizer_state_dict'],
                                scheduler_state=checkpoint['scheduler_state_dict'])

        raise ValueError('Incorrect model type! Use one of the following: "SequenceTagger", "TextClassifier".')

    def find_learning_rate(self,
                           base_path: Union[Path, str],
                           file_name: str = 'learning_rate.tsv',
                           start_learning_rate: float = 1e-7,
                           end_learning_rate: float = 10,
                           iterations: int = 100,
                           mini_batch_size: int = 32,
                           stop_early: bool = True,
                           smoothing_factor: float = 0.98,
                           **kwargs
                           ) -> Path:
        best_loss = None
        moving_avg_loss = 0

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)
        learning_rate_tsv = init_output_file(base_path, file_name)

        with open(learning_rate_tsv, 'a') as f:
            f.write('ITERATION\tTIMESTAMP\tLEARNING_RATE\tTRAIN_LOSS\n')

        optimizer = self.optimizer(self.model.parameters(), lr=start_learning_rate, **kwargs)

        train_data = self.corpus.train
        random.shuffle(train_data)
        batches = [train_data[x:x + mini_batch_size] for x in range(0, len(train_data), mini_batch_size)][:iterations]

        scheduler = ExpAnnealLR(optimizer, end_learning_rate, iterations)

        model_state = self.model.state_dict()
        model_device = next(self.model.parameters()).device
        self.model.train()

        for itr, batch in enumerate(batches):
            loss = self.model.forward_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            learning_rate = scheduler.get_lr()[0]

            loss_item = loss.item()
            if itr == 0:
                best_loss = loss_item
            else:
                if smoothing_factor > 0:
                    moving_avg_loss = smoothing_factor * moving_avg_loss + (1 - smoothing_factor) * loss_item
                    loss_item = moving_avg_loss / (1 - smoothing_factor ** (itr+1))
                if loss_item < best_loss:
                    best_loss = loss

            if stop_early and (loss_item > 4 * best_loss or torch.isnan(loss)):
                log_line(log)
                log.info('loss diverged - stopping early!')
                break

            with open(learning_rate_tsv, 'a') as f:
                f.write(f'{itr}\t{datetime.datetime.now():%H:%M:%S}\t{learning_rate}\t{loss_item}\n')

        self.model.load_state_dict(model_state)
        self.model.to(model_device)

        log_line(log)
        log.info(f'learning rate finder finished - plot {learning_rate_tsv}')
        log_line(log)

        return Path(learning_rate_tsv)
