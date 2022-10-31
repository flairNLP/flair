import json
from pathlib import Path
from typing import Dict, Optional, Union

import gdown.download_folder
import PIL
from torch.utils.data import Dataset

import flair
from flair.data import BoundingBox, Corpus, FlairDataset, Sentence
from flair.datasets.base import find_train_dev_test_files
from flair.models.sequence_tagger_utils.bioes import get_spans_from_bio


class OcrJsonDataset(FlairDataset):
    def __init__(
        self,
        path_to_split_directory: Union[str, Path],
        label_type: str = "ner",
        in_memory: bool = True,
        encoding: str = "utf-8",
        load_images: bool = False,
        label_name_map: Dict[str, str] = None,
    ):
        self.in_memory = in_memory
        path_to_split_directory = Path(path_to_split_directory)
        assert path_to_split_directory.exists()

        image_dir = path_to_split_directory / "images"
        tagged_dir = path_to_split_directory / "tagged"
        self.base_path = path_to_split_directory
        assert tagged_dir.exists()
        self.file_names = sorted([p.stem for p in tagged_dir.iterdir() if p.is_file()])
        if load_images:
            assert image_dir.exists()
            self.file_names = sorted(set([p.stem for p in image_dir.iterdir() if p.is_file()]) & set(self.file_names))

        self.total_sentence_count: int = len(self.file_names)
        self.load_images = load_images
        self.label_type = label_type
        self.encoding = encoding
        self.label_name_map = label_name_map
        if in_memory:
            self.sentences = [self._load_example(file_name) for file_name in self.file_names]

    def _remap_label(self, tag):
        # remap regular tag names
        if self.label_name_map is not None:
            return self.label_name_map.get(tag, tag)  # for example, transforming 'PER' to 'person'
        return tag

    def _load_example(self, file_name: str) -> Sentence:
        data_path = self.base_path / "tagged" / f"{file_name}.json"

        with data_path.open("r", encoding=self.encoding) as f:
            data = json.load(f)
        sentence = Sentence(text=data["words"])

        for token, bbox in zip(sentence, data["bbox"]):
            token.add_metadata("bbox", BoundingBox(*bbox))

        for span_indices, score, label in get_spans_from_bio(data["labels"]):
            span = sentence[span_indices[0] : span_indices[-1] + 1]
            value = self._remap_label(label)
            if value != "O":
                span.add_label(self.label_type, value=value, score=score)

        if self.load_images:
            img_path = self.base_path / "images" / f"{file_name}.jpg"
            with PIL.Image.open(img_path) as img:
                img.load()
            sentence.add_metadata("image", img)
        return sentence

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self) -> int:
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.in_memory:
            sentence = self.sentences[index]

        # else skip to position in file where sentence begins
        else:
            sentence = self._load_example(self.file_names[index])

            # set sentence context using partials TODO: pointer to dataset is really inefficient
            sentence._position_in_dataset = (self, index)

        return sentence


class OcrCorpus(Corpus):
    def __init__(
        self,
        train_path: Optional[Path] = None,
        dev_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        encoding: str = "utf-8",
        label_type: str = "ner",
        in_memory: bool = True,
        load_images: bool = False,
        label_name_map: Dict[str, str] = None,
        **corpusargs,
    ):
        train: Optional[Dataset] = (
            OcrJsonDataset(
                train_path,
                label_type=label_type,
                encoding=encoding,
                in_memory=in_memory,
                load_images=load_images,
                label_name_map=label_name_map,
            )
            if train_path is not None
            else None
        )

        # read in dev file if exists
        dev: Optional[Dataset] = (
            OcrJsonDataset(
                dev_path,
                label_type=label_type,
                encoding=encoding,
                in_memory=in_memory,
                load_images=load_images,
                label_name_map=label_name_map,
            )
            if dev_path is not None
            else None
        )

        # read in test file if exists
        test: Optional[Dataset] = (
            OcrJsonDataset(
                test_path,
                label_type=label_type,
                encoding=encoding,
                in_memory=in_memory,
                load_images=load_images,
                label_name_map=label_name_map,
            )
            if test_path is not None
            else None
        )

        super().__init__(train, dev, test, **corpusargs)


class SROIE(OcrCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        encoding: str = "utf-8",
        label_type: str = "ner",
        in_memory: bool = True,
        load_images: bool = False,
        label_name_map: Dict[str, str] = None,
        **corpusargs,
    ):
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        dataset_name = self.__class__.__name__.lower()
        data_folder = base_path / dataset_name

        if not data_folder.exists():
            # the url is copied from https://huggingface.co/datasets/darentang/sroie/blob/main/sroie.py#L44
            url = "https://drive.google.com/uc?id=1ZyxAw1d-9UvhgNLGRvsJK4gBCMf0VpGD"
            zip_path = base_path / "sroie.zip"
            gdown.cached_download(url, str(zip_path), postprocess=gdown.extractall)
            zip_path.unlink()
        dev_path, test_path, train_path = find_train_dev_test_files(data_folder, None, None, None)
        super().__init__(
            train_path,
            dev_path,
            test_path,
            encoding=encoding,
            label_type=label_type,
            in_memory=in_memory,
            load_images=load_images,
            label_name_map=label_name_map,
            **corpusargs,
        )
