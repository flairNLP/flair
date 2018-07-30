from subprocess import run, PIPE
from typing import List

import datetime
import os
import random
import re
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
              patience: int = 3,
              checkpoint: bool = False,
              embeddings_in_memory: bool = True,
              train_with_dev: bool = False):

        evaluation_method = 'F1'
        if self.model.tag_type in ['ner', 'np', 'srl']: evaluation_method = 'span-F1'
        if self.model.tag_type in ['pos', 'upos']: evaluation_method = 'accuracy'
        print(evaluation_method)

        os.makedirs(base_path, exist_ok=True)

        loss_txt = os.path.join(base_path, "loss.txt")
        open(loss_txt, "w", encoding='utf-8').close()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer, verbose=True, factor=anneal_factor,
                                                         patience=patience)

        train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data.extend(self.corpus.dev)

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

                # anneal against train loss
                scheduler.step(current_loss)

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

                # IMPORTANT: Switch back to train mode
                self.model.train()

                # print info
                summary = '%d' % epoch + '\t({:%H:%M:%S})'.format(datetime.datetime.now()) \
                          + '\t%f\t%d\t%f\tDEV   %d\t' % (
                current_loss, scheduler.num_bad_epochs, learning_rate, dev_fp) + dev_result
                summary = summary.replace('\n', '')
                summary += '\tTEST   \t%d\t' % test_fp + test_result

                print(summary)
                with open(loss_txt, "a") as loss_file:
                    loss_file.write('%s\n' % summary)
                    loss_file.close()

                if checkpoint and scheduler.num_bad_epochs == 0:
                    self.model.save(base_path + "/checkpoint-model.pt")

            self.model.save(base_path + "/final-model.pt")

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
            print('saving model')
            self.model.save(base_path + "/final-model.pt")
            print('done')

    def evaluate(self, evaluation: List[Sentence], out_path=None, evaluation_method: str = 'F1',
                 embeddings_in_memory: bool = True):

        tp: int = 0
        fp: int = 0

        batch_no: int = 0
        mini_batch_size = 32
        batches = [evaluation[x:x + mini_batch_size] for x in
                   range(0, len(evaluation), mini_batch_size)]

        metric = Metric('')

        lines: List[str] = []

        for batch in batches:
            batch_no += 1

            self.model.embeddings.embed(batch)

            for sentence in batch:

                sentence: Sentence = sentence

                # Step 3. Run our forward pass.
                score, tag_seq = self.model.predict_scores(sentence)

                # Step 5. Compute predictions
                predicted_id = tag_seq
                for (token, pred_id) in zip(sentence.tokens, predicted_id):
                    token: Token = token
                    # get the predicted tag
                    predicted_tag = self.model.tag_dictionary.get_item_for_index(pred_id)
                    token.add_tag('predicted', predicted_tag)

                    # get the gold tag
                    gold_tag = token.get_tag(self.model.tag_type)

                    # append both to file for evaluation
                    eval_line = token.text + ' ' + gold_tag + ' ' + predicted_tag + "\n"
                    if gold_tag == predicted_tag:
                        tp += 1
                    else:
                        fp += 1

                    lines.append(eval_line)

                lines.append('\n')

            if not embeddings_in_memory:
                self.clear_embeddings_in_batch(batch)

        if out_path != None:
            test_tsv = os.path.join(out_path, "test.tsv")
            with open(test_tsv, "w", encoding='utf-8') as outfile:
                outfile.write(''.join(lines))

        if evaluation_method == 'span-F1':
            eval_script = 'resources/tasks/eval_script'

            eval_data = ''.join(lines)

            p = run(eval_script, stdout=PIPE, input=eval_data, encoding='utf-8')
            main_result = p.stdout
            print(main_result)

            main_result = main_result.split('\n')[1]

            # parse the result file
            main_result = re.sub(';', ' ', main_result)
            main_result = re.sub('precision', 'p', main_result)
            main_result = re.sub('recall', 'r', main_result)
            main_result = re.sub('accuracy', 'acc', main_result)

            f_score = float(re.findall(r'\d+\.\d+$', main_result)[0])
            return f_score, fp, main_result

        if evaluation_method == 'accuracy':
            score = metric.accuracy()
            accuracy: float = tp / (tp + fp)
            print(accuracy)
            return score, fp, str(score)

        if evaluation_method == 'F1':
            print(metric.accuracy())
            score = metric.f_score()
            return score, fp, str(metric)

    def clear_embeddings_in_batch(self, batch: List[Sentence]):
        for sentence in batch:
            for token in sentence.tokens:
                token.clear_embeddings()
