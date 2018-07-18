from .data import Sentence, Token, TaggedCorpus, Dictionary
from .tagging_model import SequenceTagger

from typing import List, Dict, Tuple

from subprocess import run, PIPE

import torch, random, datetime, re, sys, os, shutil


class TagTrain:
    def __init__(self, model: SequenceTagger, corpus: TaggedCorpus, test_mode: bool = False) -> None:
        self.model: SequenceTagger = model
        self.corpus: TaggedCorpus = corpus
        self.test_mode: bool = test_mode

    def train(self,
              base_path: str,
              learning_rate: float = 0.1,
              mini_batch_size: int = 32,
              max_epochs: int = 100,
              save_model: bool = True,
              embeddings_in_memory: bool = True,
              train_with_dev: bool = False,
              anneal_mode: bool = False):

        checkpoint: bool = False

        evaluate_with_fscore: bool = True
        if self.model.tag_type not in ['ner', 'np', 'srl']: evaluate_with_fscore = False

        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

        loss_txt = os.path.join(self.base_path, "loss.txt")
        open(loss_txt, "w", encoding='utf-8').close()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data.extend(self.corpus.dev)

        # At any point you can hit Ctrl + C to break out of training early.
        try:

            # record overall best dev scores and best loss
            best_score = 0
            if train_with_dev: best_score = 10000
            # best_dev_score = 0
            # best_loss: float = 10000

            # this variable is used for annealing schemes
            epochs_without_improvement: int = 0

            for epoch in range(0, max_epochs):

                current_loss: int = 0

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

                # IMPORTANT: Switch to eval mode
                self.model.eval()

                if not train_with_dev:
                    print('.. evaluating... dev... ')
                    dev_score, dev_fp, dev_result = self.evaluate(self.corpus.dev,
                                                                  evaluate_with_fscore=evaluate_with_fscore,
                                                                  embeddings_in_memory=embeddings_in_memory)
                else:
                    dev_fp = 0
                    dev_result = '_'

                print('test... ')
                test_score, test_fp, test_result = self.evaluate(self.corpus.test,
                                                                 evaluate_with_fscore=evaluate_with_fscore,
                                                                 embeddings_in_memory=embeddings_in_memory)

                # IMPORTANT: Switch back to train mode
                self.model.train()

                # checkpoint model
                self.model.trained_epochs = epoch

                # is this the best model so far?
                is_best_model_so_far: bool = False

                # if dev data is used for model selection, use dev F1 score to determine best model
                if not train_with_dev and dev_score > best_score:
                    best_score = dev_score
                    is_best_model_so_far = True

                # if dev data is used for training, use training loss to determine best model
                if train_with_dev and current_loss < best_score:
                    best_score = current_loss
                    is_best_model_so_far = True

                if is_best_model_so_far:

                    print('after %d - new best score: %f' % (epochs_without_improvement, best_score))

                    epochs_without_improvement = 0

                    # save model
                    if save_model or (anneal_mode and checkpoint):
                        self.model.save(base_path + "/model.pt")
                        print('.. model saved ... ')

                else:
                    epochs_without_improvement += 1

                # anneal after 3 epochs of no improvement if anneal mode
                if epochs_without_improvement == 3 and anneal_mode:
                    best_score = current_loss
                    learning_rate /= 2

                    if checkpoint:
                        self.model = SequenceTagger.load_from_file(base_path + '/model.pt')

                    optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

                # print info
                summary = '%d' % epoch + '\t({:%H:%M:%S})'.format(datetime.datetime.now()) \
                          + '\t%f\t%d\t%f\tDEV   %d\t' % (current_loss, epochs_without_improvement, learning_rate, dev_fp) + dev_result
                summary = summary.replace('\n', '')
                summary += '\tTEST   \t%d\t' % test_fp + test_result

                print(summary)
                with open(loss_txt, "a") as loss_file:
                    loss_file.write('%s\n' % summary)
                    loss_file.close()

            self.model.save(base_path + "/final-model.pt")

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
            print('saving model')
            with open(base_path + "/final-model.pt", 'wb') as model_save_file:
                torch.save(self.model, model_save_file, pickle_protocol=4)
                model_save_file.close()
            print('done')

    def evaluate(self, evaluation: List[Sentence], evaluate_with_fscore: bool = True,
                 embeddings_in_memory: bool = True):
        tp: int = 0
        fp: int = 0

        batch_no: int = 0
        mini_batch_size = 32
        batches = [evaluation[x:x + mini_batch_size] for x in
                   range(0, len(evaluation), mini_batch_size)]

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
                    # print(token)
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

        test_tsv = os.path.join(self.base_path, "test.tsv")
        with open(test_tsv, "w", encoding='utf-8') as outfile:
            outfile.write(''.join(lines))

        if evaluate_with_fscore:
            eval_script = 'resources/tasks/eval_script'

            eval_data = ''.join(lines)

            p = run(eval_script, stdout=PIPE, input=eval_data, encoding='utf-8')
            print(p.returncode)
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

        precision: float = tp / (tp + fp)

        return precision, fp, str(precision)

    def clear_embeddings_in_batch(self, batch: List[Sentence]):
        for sentence in batch:
            for token in sentence.tokens:
                token.clear_embeddings()
