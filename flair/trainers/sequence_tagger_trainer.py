from typing import List

import datetime
import os
import random
import logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from flair.models.sequence_tagger_model import SequenceTagger
from flair.data import Sentence, Token, TaggedCorpus, Label
from flair.training_utils import Metric, init_output_file, WeightExtractor

log = logging.getLogger(__name__)


class SequenceTaggerTrainer:
    def __init__(self, model: SequenceTagger, corpus: TaggedCorpus, test_mode: bool = False) -> None:
        self.model: SequenceTagger = model
        self.corpus: TaggedCorpus = corpus
        self.test_mode: bool = test_mode

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
              ):

        evaluation_method = 'F1'
        if self.model.tag_type in ['pos', 'upos']: evaluation_method = 'accuracy'
        log.info('Evaluation method: {}'.format(evaluation_method))

        loss_txt = init_output_file(base_path, 'loss.tsv')
        with open(loss_txt, 'a') as f:
            f.write('EPOCH\tTIMESTAMP\tTRAIN_LOSS\t{}\tDEV_LOSS\t{}\tTEST_LOSS\t{}\n'.format(
                Metric.tsv_header('TRAIN'), Metric.tsv_header('DEV'), Metric.tsv_header('TEST')))

        weight_extractor = WeightExtractor(base_path)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # annealing scheduler
        anneal_mode = 'min' if train_with_dev else 'max'
        scheduler = ReduceLROnPlateau(optimizer, factor=anneal_factor, patience=patience, mode=anneal_mode, verbose=True)

        train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data.extend(self.corpus.dev)

        # At any point you can hit Ctrl + C to break out of training early.
        try:

            previous_learning_rate = learning_rate

            for epoch in range(0, max_epochs):
                log.info('-' * 100)

                bad_epochs = scheduler.num_bad_epochs
                for group in optimizer.param_groups:
                    learning_rate = group['lr']

                # reload last best model if annealing with restarts is enabled
                if learning_rate != previous_learning_rate and anneal_with_restarts and \
                        os.path.exists(base_path + "/best-model.pt"):
                    log.info('resetting to best model')
                    self.model.load_from_file(base_path + "/best-model.pt")

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if learning_rate < 0.001:
                    log.info('learning rate too small - quitting training!')
                    break

                if not self.test_mode:
                    random.shuffle(train_data)

                batches = [train_data[x:x + mini_batch_size] for x in
                           range(0, len(train_data), mini_batch_size)]

                self.model.train()

                current_loss: float = 0
                seen_sentences = 0
                modulo = max(1, int(len(batches) / 10))

                for batch_no, batch in enumerate(batches):
                    batch: List[Sentence] = batch

                    optimizer.zero_grad()

                    # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
                    loss = self.model.neg_log_likelihood(batch)

                    current_loss += loss.item()
                    seen_sentences += len(batch)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    if not embeddings_in_memory:
                        self.clear_embeddings_in_batch(batch)

                    if batch_no % modulo == 0:
                        log.info("epoch {0} - iter {1}/{2} - loss {3:.8f}".format(
                            epoch + 1, batch_no, len(batches), current_loss / seen_sentences))
                        iteration = epoch * len(batches) + batch_no
                        weight_extractor.extract_weights(self.model.state_dict(), iteration)

                current_loss /= len(train_data)

                # switch to eval mode
                self.model.eval()

                # if checkpointing is enable, save model at each epoch
                if checkpoint:
                    self.model.save(base_path + "/checkpoint.pt")

                log.info('-' * 100)

                dev_score = dev_metric = None
                if not train_with_dev:
                    dev_score, dev_metric = self.evaluate(self.corpus.dev, base_path,
                                                          evaluation_method=evaluation_method,
                                                          embeddings_in_memory=embeddings_in_memory)

                test_score, test_metric = self.evaluate(self.corpus.test, base_path,
                                                        evaluation_method=evaluation_method,
                                                        embeddings_in_memory=embeddings_in_memory)

                # anneal against train loss if training with dev, otherwise anneal against dev score
                scheduler.step(current_loss) if train_with_dev else scheduler.step(dev_score)

                # logging info
                log.info("EPOCH {0}: lr {1:.4f} - bad epochs {2}".format(epoch + 1, learning_rate, bad_epochs))
                if not train_with_dev:
                    log.info("{0:<4}: f-score {1:.4f} - acc {2:.4f} - tp {3} - fp {4} - fn {5} - tn {6}".format(
                        'DEV', dev_metric.f_score(), dev_metric.accuracy(), dev_metric._tp, dev_metric._fp,
                        dev_metric._fn, dev_metric._tn))
                log.info("{0:<4}: f-score {1:.4f} - acc {2:.4f} - tp {3} - fp {4} - fn {5} - tn {6}".format(
                    'TEST', test_metric.f_score(), test_metric.accuracy(), test_metric._tp, test_metric._fp,
                    test_metric._fn, test_metric._tn))

                with open(loss_txt, 'a') as f:
                    dev_metric_str = dev_metric.to_tsv() if dev_metric is not None else Metric.to_empty_tsv()
                    f.write('{}\t{:%H:%M:%S}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        epoch, datetime.datetime.now(), '_', Metric.to_empty_tsv(), '_', dev_metric_str, '_',
                        test_metric.to_tsv()))

                # if we use dev data, remember best model based on dev evaluation score
                if not train_with_dev and dev_score == scheduler.best:
                    self.model.save(base_path + "/best-model.pt")

            # if we do not use dev data for model selection, save final model
            if save_final_model:
                self.model.save(base_path + "/final-model.pt")

        except KeyboardInterrupt:
            log.info('-' * 100)
            log.info('Exiting from training early.')
            log.info('Saving model ...')
            self.model.save(base_path + "/final-model.pt")
            log.info('Done.')

    def evaluate(self, evaluation: List[Sentence], out_path=None, evaluation_method: str = 'F1',
                 eval_batch_size: int = 32,
                 embeddings_in_memory: bool = True):

        batch_no: int = 0
        batches = [evaluation[x:x + eval_batch_size] for x in
                   range(0, len(evaluation), eval_batch_size)]

        metric = Metric('')

        lines: List[str] = []

        for batch in batches:
            batch_no += 1

            scores, tag_seq = self.model._predict_scores_batch(batch)
            predicted_ids = tag_seq
            all_tokens = []
            for sentence in batch:
                all_tokens.extend(sentence.tokens)

            for (token, score, predicted_id) in zip(all_tokens, scores, predicted_ids):
                token: Token = token
                # get the predicted tag
                predicted_value = self.model.tag_dictionary.get_item_for_index(predicted_id)
                token.add_tag('predicted', predicted_value, score)

            for sentence in batch:

                # add predicted tags
                for token in sentence.tokens:
                    predicted_tag: Label = token.get_tag('predicted')

                    # append both to file for evaluation
                    eval_line = '{} {} {}\n'.format(token.text,
                                                    token.get_tag(self.model.tag_type).value,
                                                    predicted_tag.value)

                    lines.append(eval_line)
                lines.append('\n')

                # make list of gold tags
                gold_tags = [str(tag) for tag in sentence.get_spans(self.model.tag_type)]

                # make list of predicted tags
                predicted_tags = [str(tag) for tag in sentence.get_spans('predicted')]

                # check for true positives, false positives and false negatives
                for prediction in predicted_tags:
                    if prediction in gold_tags:
                        metric.tp()
                    else:
                        metric.fp()

                for gold in gold_tags:
                    if gold not in predicted_tags:
                        metric.fn()

            if not embeddings_in_memory:
                self.clear_embeddings_in_batch(batch)

        if out_path is not None:
            test_tsv = os.path.join(out_path, "test.tsv")
            with open(test_tsv, "w", encoding='utf-8') as outfile:
                outfile.write(''.join(lines))

        if evaluation_method == 'accuracy':
            score = metric.accuracy()
            return score, metric

        if evaluation_method == 'F1':
            score = metric.f_score()
            return score, metric

    def clear_embeddings_in_batch(self, batch: List[Sentence]):
        for sentence in batch:
            for token in sentence.tokens:
                token.clear_embeddings()
