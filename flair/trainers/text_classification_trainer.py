import random
from collections import defaultdict
from functools import reduce
from typing import List

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from flair.data import Sentence, TaggedCorpus, Dictionary
from flair.models.text_classification_model import TextClassifier
from flair.training_utils import convert_labels_to_one_hot, calculate_micro_avg_metric, init_output_file, clear_embeddings, \
    calculate_class_metrics


class TextClassifierTrainer:
    """
    Training class to train and evaluate a text classification model.
    """

    def __init__(self, model: TextClassifier, corpus: TaggedCorpus, label_dict: Dictionary, test_mode: bool = False) -> None:
        self.model: TextClassifier = model
        self.corpus: TaggedCorpus = corpus
        self.label_dict: Dictionary = label_dict
        self.test_mode: bool = test_mode

    def train(self,
              base_path: str,
              learning_rate: float = 0.1,
              mini_batch_size: int = 32,
              max_epochs: int = 100,
              anneal_factor: float = 0.5,
              patience: int = 2,
              save_model: bool = True,
              embeddings_in_memory: bool = True,
              train_with_dev: bool = False):
        """
        Trains the model using the training data of the corpus.
        :param base_path: the directory to which any results should be written to
        :param learning_rate: the learning rate
        :param mini_batch_size: the mini batch size
        :param max_epochs: the maximum number of epochs to train
        :param save_model: boolean value indicating, whether the model should be saved or not
        :param embeddings_in_memory: boolean value indicating, if embeddings should be kept in memory or not
        :param train_with_dev: boolean value indicating, if the dev data set should be used for training or not
        """

        loss_txt = init_output_file(base_path, 'loss.txt')
        with open(loss_txt, 'a') as f:
            f.write('EPOCH\tITERATION\tDEV_LOSS\tTRAIN_LOSS\tDEV_F_SCORE\tTRAIN_F_SCORE\tDEV_ACC\tTRAIN_ACC\n')
        weights_txt = init_output_file(base_path, 'weights.txt')

        weights_index = defaultdict(lambda: defaultdict(lambda: list()))

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
            # record overall best dev scores and best loss
            best_score = 0

            for epoch in range(max_epochs):
                print('-' * 100)
                if not self.test_mode:
                    random.shuffle(train_data)

                batches = [train_data[x:x + mini_batch_size] for x in range(0, len(train_data), mini_batch_size)]

                current_loss: float = 0
                seen_sentences = 0
                modulo = max(1, int(len(batches) / 10))

                self.model.train()

                for batch_no, batch in enumerate(batches):
                    scores = self.model.forward(batch)
                    loss = self.model.calculate_loss(scores, batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    seen_sentences += len(batch)
                    current_loss += loss.item()

                    if not embeddings_in_memory:
                        clear_embeddings(batch)

                    if batch_no % modulo == 0:
                        print("epoch {0} - iter {1}/{2} - loss {3:.8f}".format(epoch + 1, batch_no, len(batches),
                                                                               current_loss / seen_sentences))

                        iteration = epoch * len(batches) + batch_no
                        self._extract_weigths(iteration, weights_index, weights_txt)

                current_loss /= len(train_data)

                # IMPORTANT: Switch to eval mode
                self.model.eval()

                print('-' * 100)
                train_metrics, train_loss = self.evaluate(self.corpus.train, mini_batch_size=mini_batch_size,
                                                          embeddings_in_memory=embeddings_in_memory)
                train_f_score = train_metrics['MICRO_AVG'].f_score()
                train_acc = train_metrics['MICRO_AVG'].accuracy()
                print("{0:<7} epoch {1} - loss {2:.8f} - f-score {3:.4f} - acc {4:.4f}".format(
                    'TRAIN:', epoch, train_loss, train_f_score, train_acc))

                dev_f_score = dev_acc = dev_loss = 0
                if not train_with_dev:
                    dev_metrics, dev_loss = self.evaluate(self.corpus.dev, mini_batch_size=mini_batch_size,
                                                          embeddings_in_memory=embeddings_in_memory)
                    dev_f_score = dev_metrics['MICRO_AVG'].f_score()
                    dev_acc = dev_metrics['MICRO_AVG'].accuracy()
                    print("{0:<7} epoch {1} - loss {2:.8f} - f-score {3:.4f} - acc {4:.4f}".format(
                        'DEV:', epoch, dev_loss, dev_f_score, dev_acc))

                with open(loss_txt, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        epoch, epoch * len(batches), dev_loss, train_loss, dev_f_score, train_f_score, dev_acc, train_acc))

                # IMPORTANT: Switch back to train mode
                self.model.train()

                # anneal against train loss if training with dev, otherwise anneal against dev score
                scheduler.step(current_loss) if train_with_dev else scheduler.step(dev_f_score)

                is_best_model_so_far: bool = False
                current_score = dev_f_score if not train_with_dev else train_f_score

                if current_score > best_score:
                    best_score = current_score
                    is_best_model_so_far = True

                if is_best_model_so_far:
                    if save_model:
                        self.model.save(base_path + "/model.pt")

            self.model.save(base_path + "/final-model.pt")

            if save_model:
                self.model = TextClassifier.load_from_file(base_path + "/model.pt")

            print('-' * 100)
            print('testing...')

            test_metrics, test_loss = self.evaluate(
                self.corpus.test, mini_batch_size=mini_batch_size, eval_class_metrics=True,
                embeddings_in_memory=embeddings_in_memory)

            for metric in test_metrics.values():
                metric.print()

            print('-' * 100)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
            print('saving model')
            with open(base_path + "/final-model.pt", 'wb') as model_save_file:
                torch.save(self.model, model_save_file, pickle_protocol=4)
                model_save_file.close()
            print('done')

    def evaluate(self, sentences: List[Sentence], eval_class_metrics: bool = False, mini_batch_size: int = 32,
                 embeddings_in_memory: bool = True) -> (dict, float):
        """
        Evaluates the model with the given list of sentences.
        :param sentences: the list of sentences
        :param mini_batch_size: the mini batch size to use
        :return: list of metrics, and the loss
        """
        eval_loss = 0

        batches = [sentences[x:x + mini_batch_size] for x in
                   range(0, len(sentences), mini_batch_size)]

        y_pred = []
        y_true = []

        for batch in batches:
            scores = self.model.forward(batch)
            labels = self.model.obtain_labels(scores)
            loss = self.model.calculate_loss(scores, batch)

            eval_loss += loss

            y_true.extend([sentence.get_label_names() for sentence in batch])
            y_pred.extend([[label.name for label in sent_labels] for sent_labels in labels])

            if not embeddings_in_memory:
                clear_embeddings(batch)

        y_pred = convert_labels_to_one_hot(y_pred, self.label_dict)
        y_true = convert_labels_to_one_hot(y_true, self.label_dict)

        metrics = [calculate_micro_avg_metric(y_true, y_pred, self.label_dict)]
        if eval_class_metrics:
            metrics.extend(calculate_class_metrics(y_true, y_pred, self.label_dict))

        eval_loss /= len(sentences)

        metrics_dict = {metric.name: metric for metric in metrics}

        return metrics_dict, eval_loss

    def _extract_weigths(self, iteration, weights_index, weights_txt):
        for key in self.model.state_dict().keys():

            vec = self.model.state_dict()[key]
            weights_to_watch = min(10, reduce(lambda x, y: x*y, list(vec.size())))

            if key not in weights_index:
                self._init_weights_index(key, weights_index, weights_to_watch)

            for i in range(weights_to_watch):
                vec = self.model.state_dict()[key]
                for index in weights_index[key][i]:
                    vec = vec[index]

                value = vec.item()

                with open(weights_txt, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\n'.format(iteration, key, i, float(value)))

    def _init_weights_index(self, key, weights_index, weights_to_watch):
        indices = {}

        i = 0
        while len(indices) < weights_to_watch:
            vec = self.model.state_dict()[key]
            cur_indices = []

            for x in range(len(vec.size())):
                index = random.randint(0, len(vec) - 1)
                vec = vec[index]
                cur_indices.append(index)

            if cur_indices not in list(indices.values()):
                indices[i] = cur_indices
                i += 1

        weights_index[key] = indices