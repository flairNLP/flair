import json
import logging
import os
import urllib
from pathlib import Path

import numpy as np
import torch.utils.data.dataloader
from torch.utils.data import Dataset
from tqdm import tqdm

from flair.data import Corpus, DataPair, FlairDataset, Image, Sentence
from flair.file_utils import cached_path

log = logging.getLogger("flair")


class FeideggerCorpus(Corpus):
    def __init__(self, **kwargs) -> None:
        dataset = "feidegger"

        # cache Feidegger config file
        json_link = "https://raw.githubusercontent.com/zalandoresearch/feidegger/master/data/FEIDEGGER_release_1.1.json"
        json_local_path = cached_path(json_link, Path("datasets") / dataset)

        # cache Feidegger images
        with json_local_path.open(encoding="utf-8") as fin:
            dataset_info = json.load(fin)
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

        train_indices = list(np.where(np.isin(feidegger_dataset.split, list(range(8))))[0])  # type: ignore[attr-defined]
        train = torch.utils.data.dataset.Subset(feidegger_dataset, train_indices)

        dev_indices = list(np.where(np.isin(feidegger_dataset.split, [8]))[0])  # type: ignore[attr-defined]
        dev = torch.utils.data.dataset.Subset(feidegger_dataset, dev_indices)

        test_indices = list(np.where(np.isin(feidegger_dataset.split, [9]))[0])  # type: ignore[attr-defined]
        test = torch.utils.data.dataset.Subset(feidegger_dataset, test_indices)

        super().__init__(train, dev, test, name="feidegger")


class FeideggerDataset(FlairDataset):
    def __init__(self, dataset_info, **kwargs) -> None:
        super().__init__()

        self.data_points: list[DataPair] = []
        self.split: list[int] = []

        def identity(x):
            return x

        preprocessor = identity
        if kwargs.get("lowercase"):
            preprocessor = str.lower

        for image_info in dataset_info:
            image = Image(imageURL=image_info["url"])
            for caption in image_info["descriptions"]:
                # append Sentence-Image data point
                self.data_points.append(DataPair(Sentence(preprocessor(caption), use_tokenizer=True), image))
                self.split.append(int(image_info["split"]))

    def __len__(self) -> int:
        return len(self.data_points)

    def __getitem__(self, index: int = 0) -> DataPair:
        return self.data_points[index]

    def is_in_memory(self) -> bool:
        return True
