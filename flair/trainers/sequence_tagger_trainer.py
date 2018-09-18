import copy
from typing import List

import datetime
import os
import random
import sys
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from flair.models.sequence_tagger_model import SequenceTagger
from flair.data import Sentence, Token, TaggedCorpus
from flair.training_utils import Metric


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
              patience: int = 2,
              train_with_dev: bool = False,
              embeddings_in_memory: bool = True,
              checkpoint: bool = False,
              save_final_model: bool = True,
              ):

        evaluation_method = 'F1'
        if self.model.tag_type in ['pos', 'upos']: evaluation_method = 'accuracy'
        print(evaluation_method)

        os.makedirs(base_path, exist_ok=True)

        loss_txt = os.path.join(base_path, "loss.txt")
        open(loss_txt, "w", encoding='utf-8').close()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        anneal_mode = 'min' if train_with_dev else 'max'
        scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer, factor=anneal_factor, patience=patience,
                                                         mode=anneal_mode)

        train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data.extend(self.corpus.dev)

        # variable to store best model during training
        best_model = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:

            for epoch in range(0, max_epochs):

                current_loss: int = 0

                for group in optimizer.param_groups:
                    learning_rate = group['lr']

                if not self.test_mode: random.shuffle(train_data)

                batches = [train_data[x:x + mini_batch_size] for x in range(0, len(train_data), mini_batch_size)]

                self.model.train()

                batch_no: int = 0

                for batch in batches:
                    batch: List[Sentence] = batch
                    batch_no += 1

                    if batch_no % 100 == 0:
                        print("%d of %d (%f)" % (batch_no, len(batches), float(batch_no / len(batches))))

                    optimizer.zero_grad()

                    # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
                    loss = self.model.neg_log_likelihood(batch, self.model.tag_type)

                    current_loss += loss.item()

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                    optimizer.step()

                    sys.stdout.write('.')
                    sys.stdout.flush()

                    if not embeddings_in_memory:
                        self.clear_embeddings_in_batch(batch)

                current_loss /= len(train_data)

                # switch to eval mode
                self.model.eval()

                if not train_with_dev:
                    print('.. evaluating... dev... ')
                    dev_score, dev_fp, dev_result = self.evaluate(self.corpus.dev, base_path,
                                                                  evaluation_method=evaluation_method,
                                                                  embeddings_in_memory=embeddings_in_memory)
                else:
                    dev_fp = 0
                    dev_result = '_'

                print('test... ')
                test_score, test_fp, test_result = self.evaluate(self.corpus.test, base_path,
                                                                 evaluation_method=evaluation_method,
                                                                 embeddings_in_memory=embeddings_in_memory)

                # anneal against train loss if training with dev, otherwise anneal against dev score
                scheduler.step(current_loss) if train_with_dev else scheduler.step(dev_score)

                summary = '%d' % epoch + '\t({:%H:%M:%S})'.format(datetime.datetime.now()) \
                          + '\t%f\t%d\t%f\tDEV   %d\t' % (
                              current_loss, scheduler.num_bad_epochs, learning_rate, dev_fp) + dev_result
                summary = summary.replace('\n', '')
                summary += '\tTEST   \t%d\t' % test_fp + test_result

                print(summary)
                with open(loss_txt, "a") as loss_file:
                    loss_file.write('%s\n' % summary)
                    loss_file.close()

                if checkpoint:
                    self.model.save(base_path + "/checkpoint.pt")

                # if we use dev data, remember best model based on dev evaluation score
                if not train_with_dev and dev_score == scheduler.best:
                    best_model = copy.deepcopy(self.model)

            # if we do not use dev data for model selection, save final model
            if save_final_model:
                if train_with_dev:
                    self.model.save(base_path + "/final-model.pt")
                else:
                    best_model.save(base_path + "/best-model.pt")

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
            print('saving model')
            self.model.save(base_path + "/final-model.pt")
            print('done')

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

            score, tag_seq = self.model._predict_scores_batch(batch)
            predicted_id = tag_seq
            all_tokens = []
            for sentence in batch:
                all_tokens.extend(sentence.tokens)

            for (token, pred_id) in zip(all_tokens, predicted_id):
                token: Token = token
                # get the predicted tag
                predicted_tag = self.model.tag_dictionary.get_item_for_index(pred_id)
                token.add_tag('predicted', predicted_tag)

            for sentence in batch:

                sentence: Sentence = sentence

                # add predicted tags
                for (token, pred_id) in zip(sentence.tokens, tag_seq):
                    # append both to file for evaluation
                    eval_line = '{} {} {}\n'.format(token.text,
                                                    token.get_tag(self.model.tag_type),
                                                    token.get_tag('predicted'))
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
            return score, metric._fp, str(score)

        if evaluation_method == 'F1':
            score = metric.f_score()
            return score, metric._fp, str(metric)

    def clear_embeddings_in_batch(self, batch: List[Sentence]):
        for sentence in batch:
            for token in sentence.tokens:
                token.clear_embeddings()
