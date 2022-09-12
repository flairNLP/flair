import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import numpy as np


torch.manual_seed(42)


class GazetteerFeaturesDataset(Dataset):
    def __init__(self, gazetteer_path,
                 random_spans_path,
                 downsample=1.0,
                 balance_random=True,
                 factor_random_true=1,
                 min_tagged_freq=20,
                 device=torch.device("cpu")):

        self.gazetteer = pd.read_csv(gazetteer_path)
        self.random_spans = pd.read_csv(random_spans_path)
        self.factor_random_true = factor_random_true
        self.balance_random = balance_random
        self.downsample = downsample
        self.min_tagged_freq = min_tagged_freq
        self.device = device

        self.gazetteer = self.gazetteer.sample(frac=self.downsample)
        print("Originally: \t gazetteer len:", len(self.gazetteer), "random_spans len:", len(self.random_spans))

        # filter out those with <= min_tagged_freq
        self.gazetteer = self.gazetteer.drop(self.gazetteer[self.gazetteer.tagged_frequency < self.min_tagged_freq].index)

        if self.balance_random and len(self.random_spans) > self.factor_random_true*len(self.gazetteer):
            self.random_spans = self.random_spans.sample(self.factor_random_true*len(self.gazetteer))

        print("After: \t gazetteer len:", len(self.gazetteer), "random_spans len:", len(self.random_spans))

        self.joined = pd.concat([self.gazetteer, self.random_spans])

        print(self.joined.head())

    def string_transform(self, span_string):
        return str(span_string)

    def target_transform(self, vector):
        cleaned_vector = np.array(vector[0:4])  # just the MISC, ORG, PER, LOC counts
        observation_count = np.array(vector[4])  # abs_span_freq
        sum_tagged = np.sum(vector[0:4])  # sum of MISC, ORG, PER, LOC
        tagged_ratio = np.array(vector[6]).reshape((1,))  # this is precomputed in gaz file

        DEFINED_MAX = 50  # TODO: what to choose here? make parameter
        confidence = np.array([min(sum_tagged / DEFINED_MAX, 1)])
        # print((cleaned_vector / sum_tagged).shape, confidence.shape, tagged_ratio.shape)

        if sum_tagged > 0:
            rt_vector = (np.concatenate((cleaned_vector / sum_tagged, confidence, tagged_ratio), 0))
        else:
            rt_vector = (np.concatenate((cleaned_vector, confidence, tagged_ratio), 0))

        features = list(rt_vector[:-2])
        #weighting_info = list(rt_vector[-2:])
        # rt_vector = np.around(rt_vector, decimals = 5)
        return features#, weighting_info

    def __len__(self):
        return len(self.joined)

    def __getitem__(self, idx):
        span_string = self.joined.iloc[idx, 0]
        vector = self.joined.iloc[idx,1:]

        span_string = self.string_transform(span_string)
        features = self.target_transform(vector)
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        #weighting_info = torch.tensor(weighting_info, dtype=torch.float32).to(self.device)

        return span_string, features


def get_dataloaders(gazetteer_path,
                    random_spans_path,
                    batch_size = 32,
                    device = torch.device("cuda:1"),
                    downsample = 1.0,
                    min_tagged_freq = 20,
                    factor_random_true = 1):

    gazetteer_dataset = GazetteerFeaturesDataset(gazetteer_path, random_spans_path,
                                                 downsample = downsample,
                                                 device=device, factor_random_true=factor_random_true,
                                                 min_tagged_freq=min_tagged_freq)

    print(len(gazetteer_dataset))

    train_size = int(0.995 * len(gazetteer_dataset))
    val_size = len(gazetteer_dataset) - train_size
    dev_size = int(0.5 * val_size)
    test_size = val_size - dev_size

    train_set, dev_set, test_set = torch.utils.data.random_split(gazetteer_dataset, [train_size, dev_size, test_size])

    print(len(train_set), len(dev_set), len(test_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=None, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, collate_fn=None, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=None, shuffle=True)

    #train_loader = FlairDataLoader(train_set, batch_size=batch_size, shuffle=True)
    #dev_loader = FlairDataLoader(dev_set, batch_size=batch_size, shuffle=True)
    #test_loader = FlairDataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, dev_loader, test_loader



# class GazetteerFeaturesFlairDataset(FlairDataset):
#     def __init__(self, gazetteer_path, random_spans_path,
#                  downsample=1.0, balance_random=True, factor_random_true=1,
#                  use_tokenizer: Union[bool, Tokenizer] = SpaceTokenizer(),
#                  device=torch.device("cuda:1")):
#         super().__init__()
#         self.gazetteer = pd.read_csv(gazetteer_path)
#         self.random_spans = pd.read_csv(random_spans_path)
#         self.factor_random_true = factor_random_true
#         self.balance_random = balance_random
#         self.downsample = downsample
#         self.device = device
#
#         self.gazetteer = self.gazetteer.sample(frac=self.downsample)
#
#         if self.balance_random and len(self.random_spans) > self.factor_random_true * len(self.gazetteer):
#             self.random_spans = self.random_spans.sample(self.factor_random_true * len(self.gazetteer))
#
#         print("gazetteer len:", len(self.gazetteer), "random_spans len:", len(self.random_spans))
#
#         self.joined = pd.concat([self.gazetteer, self.random_spans])
#
#         print(self.joined.head())
#
#         self.use_tokenizer = use_tokenizer
#         self.texts = [Sentence(str(t), use_tokenizer=self.use_tokenizer) for t in self.joined.span_string.values ]
#         self.targets = self.joined.iloc[:, 1:].values
#
#         del self.gazetteer
#         del self.random_spans
#
#     def is_in_memory(self) -> bool:
#         return True
#
#     def __len__(self):
#         return len(self.texts)
#
#     def target_transform(self, vector):
#         cleaned_vector = np.array(vector[0:4])  # just the MISC, ORG, PER, LOC counts
#         observation_count = np.array(vector[4])  # abs_span_freq
#         sum_tagged = np.sum(vector[0:4])  # sum of MISC, ORG, PER, LOC
#         tagged_ratio = np.array(vector[6]).reshape((1,))  # this is precomputed in gaz file
#
#         DEFINED_MAX = 50  # TODO: what to choose here? make parameter
#         confidence = np.array([min(sum_tagged / DEFINED_MAX, 1)])
#         # print((cleaned_vector / sum_tagged).shape, confidence.shape, tagged_ratio.shape)
#
#         if sum_tagged > 0:
#             rt_vector = (np.concatenate((cleaned_vector / sum_tagged, confidence, tagged_ratio), 0))
#         else:
#             rt_vector = (np.concatenate((cleaned_vector, confidence, tagged_ratio), 0))
#
#         features = list(rt_vector[:-2]) # just use the labels
#         #weighting_info = list(rt_vector[-2:])
#         features = np.around(features, decimals = 5)
#         return features
#
#     def __getitem__(self, index: int = 0):
#         text = self.texts[index]
#         target = self.target_transform(self.targets[index])
#         return text, torch.tensor(target, dtype=torch.float32).to(self.device)
#