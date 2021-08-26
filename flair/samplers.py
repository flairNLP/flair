from typing import Optional, Tuple

import logging

import types

from collections import defaultdict
from functools import wraps

import random, torch

from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

from math import ceil, floor

from flair.data import Sentence, Corpus, FlairDataset

log = logging.getLogger("flair")


class FlairSampler(Sampler):
    def set_dataset(self, data_source):
        """Initialize by passing a block_size and a plus_window parameter.
        :param data_source: dataset to sample from
        """
        self.data_source = data_source
        self.num_samples = len(self.data_source)

    def __len__(self):
        return self.num_samples


class ImbalancedClassificationDatasetSampler(FlairSampler):
    """Use this to upsample rare classes and downsample common classes in your unbalanced classification dataset.
    """

    def __init__(self):
        super(ImbalancedClassificationDatasetSampler, self).__init__(None)

    def set_dataset(self, data_source: FlairDataset):
        """
        Initialize by passing a classification dataset with labels, i.e. either TextClassificationDataSet or
        :param data_source:
        """
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.indices = list(range(len(data_source)))

        # first determine the distribution of classes in the dataset
        label_count = defaultdict(int)
        for sentence in data_source:
            for label in sentence.labels:
                label_count[label.value] += 1

        # weight for each sample
        offset = 0
        weights = [
            1.0 / (offset + label_count[data_source[idx].labels[0].value])
            for idx in self.indices
        ]

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )


class ChunkSampler(FlairSampler):
    """Splits data into blocks and randomizes them before sampling. This causes some order of the data to be preserved,
    while still shuffling the data.
    """

    def __init__(self, block_size=5, plus_window=5):
        super(ChunkSampler, self).__init__(None)
        self.block_size = block_size
        self.plus_window = plus_window
        self.data_source = None

    def __iter__(self):
        data = list(range(len(self.data_source)))

        blocksize = self.block_size + random.randint(0, self.plus_window)

        log.info(
            f"Chunk sampling with blocksize = {blocksize} ({self.block_size} + {self.plus_window})"
        )

        # Create blocks
        blocks = [data[i : i + blocksize] for i in range(0, len(data), blocksize)]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]
        return iter(data)


class ExpandingChunkSampler(FlairSampler):
    """Splits data into blocks and randomizes them before sampling. Block size grows with each epoch.
    This causes some order of the data to be preserved, while still shuffling the data.
    """

    def __init__(self, step=3):
        """Initialize by passing a block_size and a plus_window parameter.
        :param data_source: dataset to sample from
        """
        super(ExpandingChunkSampler, self).__init__(None)
        self.block_size = 1
        self.epoch_count = 0
        self.step = step

    def __iter__(self):
        self.epoch_count += 1

        data = list(range(len(self.data_source)))

        log.info(f"Chunk sampling with blocksize = {self.block_size}")

        # Create blocks
        blocks = [
            data[i : i + self.block_size] for i in range(0, len(data), self.block_size)
        ]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]

        if self.epoch_count % self.step == 0:
            self.block_size += 1

        return iter(data)


def modify_train_data_for_token_access(train_data : FlairDataset):
    """" Monkeypatch for usual sentence based datasets.

         During training, data points are retrieved via (sentence id, token id),
         allowing the model to be trained on individual tokens instead of whole
         sentences. Sentences are still returned for embedding.
    """

    _num_sentences = len(train_data)

    def num_sentences(self):
        return _num_sentences

    # binds instance to function (function to method)
    train_data.num_sentences = types.MethodType(num_sentences, train_data)

    train_data.get_sentence = train_data.__getitem__

    total_tokens = 0

    for sentence in train_data:
        # accumulate total number of tokens
        total_tokens += len(sentence)

    def get_token(self, sentence_idx: int, token_idx : int) -> Tuple[Sentence, int]:
        sentence = train_data.get_sentence(sentence_idx)
        return sentence[token_idx]

    train_data.get_token = types.MethodType(get_token, train_data)

    class MonkeyPatchedTrainDataset(train_data.__class__):
        def __getitem__(self, indices : Tuple[int, int]):
            return self.get_token(*indices)

        def __len__(self):
            return total_tokens

    train_data.__class__ = MonkeyPatchedTrainDataset


class EpisodicSampler(FlairSampler):
    def __init__(self,
                 num_support : int,
                 num_query : int,
                 num_episodes :int,
                 num_classes : Optional[int] = None,
                 tag_type = 'frame',
                 data_source=None,
                 uniform_classes=True,
                 excluded_labels=["O"]):

        self.num_support = num_support
        self.num_query = num_query
        self.num_classes = num_classes
        self.num_episodes = num_episodes
        self.tag_type = tag_type
        self.uniform_classes = uniform_classes
        self.excluded_labels = excluded_labels

        self._dataset = None

    def set_dataset(self, train_data : FlairDataset):
        # safeguard agains double call of function (would mess with the monkey patch)
        if self._dataset:
            return
        self._dataset = train_data

        # split dataset by class
        classes = dict()

        # iter through all tokens in all sentences
        for sentence_idx in range(len(train_data)):
            sentence = train_data[sentence_idx]

            for token_idx, token in enumerate(sentence):
                cls = token.get_tag(self.tag_type).value

                if cls in self.excluded_labels:
                    continue

                classes.setdefault(cls, list()).append((sentence_idx, token_idx))

        self.buckets = list(classes.values())


        if not self.uniform_classes:
            self.class_pool = list()

            # add classes according to their number of occurences
            for i, b in enumerate(self.buckets):
                times = torch.tensor(len(b)).float().log().floor()*8 + 1
                self.class_pool += [i]*int(times)

        # modify train_data to allow direct token access
        modify_train_data_for_token_access(train_data)

    def sample_episode(self, classes):
        if classes is None:
            episode_buckets = self.buckets
        else:
            episode_buckets = [self.buckets[i] for i in classes]

        support_set = list()
        query_set = list()

        for bucket in episode_buckets:
            if len(bucket) >= (self.num_support + self.num_query):
                indices = torch.randperm(len(bucket))

                for i in indices[:self.num_support]:
                    support_set.append(bucket[i])

                for i in indices[self.num_support:self.num_support + self.num_query]:
                    query_set.append(bucket[i])

            # In the next cases, we deal with unsuffiecient numbe of samples
            # in class. Since sample indices are returned from a sampler
            # (not a batch_sampler!), they are returned one by one (flair
            # constructs its own batch_sampler). Hence, it is important
            # that the number of samples in the episode remains constant.
            # Otherwise the episode would be split up over multiple batches
            # an one batch might contain the remains and beginning of two
            # episodes.

            elif len(bucket) == 1:
                #if there is only one sample, use it as both support and query
                support_set += [bucket[0]] * self.num_support
                query_set += [bucket[0]] * self.num_query

            else:
                # if there are not enough samples, but more then one
                # split fairly, by considering support query ratio and having
                # at least one sample in each set
                support_ratio = self.num_support / (self.num_support + self.num_query)

                if self.num_support < self.num_query:
                    num_support = ceil(support_ratio * len(bucket))
                else:
                    num_support = floor(support_ratio * len(bucket))

                indices = torch.randperm(len(bucket))

                for i in range(self.num_support):
                    support_set.append(
                        bucket[indices[i % num_support]]
                    )

                num_query = len(bucket) - num_support

                for i in range(self.num_query):
                    query_set.append(
                        bucket[indices[num_support + (i % num_query)]]
                    )

        random.shuffle(query_set)

        yield from support_set
        yield from query_set

    def __iter__(self):
        if self.uniform_classes:
            for _ in range(self.num_episodes):
                if self.num_classes is None:
                    yield from self.sample_epsiode(None)
                else:
                    # sample some classes from all available classes
                    yield from self.sample_episode(
                        torch.randperm(len(self.buckets))[:self.num_classes]
                    )
        else:
            batches = DataLoader(self.class_pool,
                                 batch_size=self.num_classes,
                                 drop_last=True,
                                 shuffle=True)

            for batch in batches:
                yield from self.sample_episode(batch)

    @property
    def support_size(self):
        if self.num_classes is None:
            return len(self.buckets) * self.num_support
        else:
            return self.num_classes  * self.num_support

    @property
    def query_size(self):
        if self.num_classes is None:
            return len(self.buckets) * self.num_query
        else:
            return self.num_classes  * self.num_query

    @property
    def episode_size(self):
        return self.support_size + self.query_size

    def __len__(self):
        # This ensures that the flair model trainer consumes num_episodes
        # number of episodes and no incomplete epsiode remains at the end
        if self.uniform_classes:
            return self.num_episodes * self.episode_size
        else:
            return len(self.class_pool)  * (self.num_support + self.num_query)

    def sorted_samples(self):
        for bucket in self.buckets:
            yield (self._dataset[indices] for indices in bucket)
