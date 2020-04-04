import logging
import os
import numpy as np
import json
import urllib

from tqdm import tqdm
from pathlib import Path
from typing import List

import torch.utils.data.dataloader
from torch.utils.data import Dataset

from flair.data import (
    Sentence,
    Corpus,
    FlairDataset,
    DataPair,
    Image,
)
from flair.file_utils import cached_path

log = logging.getLogger("flair")


class FeideggerCorpus(Corpus):
    def __init__(self, **kwargs):
        dataset = "feidegger"

        # cache Feidegger config file
        json_link = "https://raw.githubusercontent.com/zalandoresearch/feidegger/master/data/FEIDEGGER_release_1.1.json"
        json_local_path = cached_path(json_link, Path("datasets") / dataset)

        # cache Feidegger images
        dataset_info = json.load(open(json_local_path, "r"))
        images_cache_folder = os.path.join(os.path.dirname(json_local_path), "images")
        if not os.path.isdir(images_cache_folder):
            os.mkdir(images_cache_folder)
        for image_info in tqdm(dataset_info):
            name = os.path.basename(image_info["url"])
            filename = os.path.join(images_cache_folder, name)
            if not os.path.isfile(filename):
                urllib.request.urlretrieve(image_info["url"], filename)
            # replace image URL with local cached file
            image_info["url"] = filename

        feidegger_dataset: Dataset = FeideggerDataset(dataset_info, **kwargs)

        train_indices = list(
            np.where(np.in1d(feidegger_dataset.split, list(range(8))))[0]
        )
        train = torch.utils.data.dataset.Subset(feidegger_dataset, train_indices)

        dev_indices = list(np.where(np.in1d(feidegger_dataset.split, [8]))[0])
        dev = torch.utils.data.dataset.Subset(feidegger_dataset, dev_indices)

        test_indices = list(np.where(np.in1d(feidegger_dataset.split, [9]))[0])
        test = torch.utils.data.dataset.Subset(feidegger_dataset, test_indices)

        super(FeideggerCorpus, self).__init__(train, dev, test, name="feidegger")


class FeideggerDataset(FlairDataset):
    def __init__(self, dataset_info, in_memory: bool = True, **kwargs):
        super(FeideggerDataset, self).__init__()

        self.data_points: List[DataPair] = []
        self.split: List[int] = []

        preprocessor = lambda x: x
        if "lowercase" in kwargs and kwargs["lowercase"]:
            preprocessor = lambda x: x.lower()

        for image_info in dataset_info:
            image = Image(imageURL=image_info["url"])
            for caption in image_info["descriptions"]:
                # append Sentence-Image data point
                self.data_points.append(
                    DataPair(Sentence(preprocessor(caption), use_tokenizer=True), image)
                )
                self.split.append(int(image_info["split"]))

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index: int = 0) -> DataPair:
        return self.data_points[index]