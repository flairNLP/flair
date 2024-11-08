import json
from pathlib import Path
from typing import Optional, Union

import gdown.download_folder
import PIL
from torch.utils.data import Dataset

import flair
from flair.data import BoundingBox, Corpus, FlairDataset, Sentence, get_spans_from_bio
from flair.datasets.base import find_train_dev_test_files


class OcrJsonDataset(FlairDataset):
    def __init__(
        self,
        path_to_split_directory: Union[str, Path],
        label_type: str = "ner",
        in_memory: bool = True,
        encoding: str = "utf-8",
        load_images: bool = False,
        normalize_coords_to_thousands: bool = True,
        label_name_map: Optional[dict[str, str]] = None,
    ) -> None:
        """Instantiates a Dataset from a OCR-Json format.

        The folder is structured with a "images" folder and a "tagged" folder.
        Those folders contain respectively .jpg and .json files with matching file name.
        The json contains 3 fields "words", "bbox", "labels" which are lists of equal length
        "words" is a list of strings, containing the ocr texts,
        "bbox" is a list of int-Tuples, containing left, top, right, bottom
        "labels" is a BIO-tagging of the sentences
        :param path_to_split_directory: base folder with the task data
        :param label_type: the label_type to add the ocr labels to
        :param encoding: the encoding to load the .json files with
        :param normalize_coords_to_thousands: if True, the coordinates will be ranged from 0 to 1000
        :param load_images: if True, the pillow images will be added as metadata
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param label_name_map: Optionally map tag names to different schema.
        :return: a Dataset with Sentences that contain OCR information
        """
        self.in_memory = in_memory
        path_to_split_directory = Path(path_to_split_directory)
        assert path_to_split_directory.exists()

        image_dir = path_to_split_directory / "images"
        tagged_dir = path_to_split_directory / "tagged"
        self.base_path = path_to_split_directory
        assert tagged_dir.exists()
        assert image_dir.exists()
        self.file_names = sorted(
            {p.stem for p in image_dir.iterdir() if p.is_file()} & {p.stem for p in tagged_dir.iterdir() if p.is_file()}
        )

        self.total_sentence_count: int = len(self.file_names)
        self.load_images = load_images
        self.label_type = label_type
        self.encoding = encoding
        self.label_name_map = label_name_map
        self.normalize_coords_to_thousands = normalize_coords_to_thousands
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

        img_path = self.base_path / "images" / f"{file_name}.jpg"
        with PIL.Image.open(img_path) as img:
            width, height = img.size
            if self.load_images:
                img.load()
                sentence.add_metadata("image", img.convert("RGB"))
            sentence.add_metadata("img_width", width)
            sentence.add_metadata("img_height", height)

        for token, (left, top, right, bottom) in zip(sentence, data["bbox"]):
            if self.normalize_coords_to_thousands:
                left = int(1000 * left / width)
                top = int(1000 * top / height)
                right = int(1000 * right / width)
                bottom = int(1000 * bottom / height)

            token.add_metadata("bbox", BoundingBox(left=left, top=top, right=right, bottom=bottom))

        for span_indices, score, label in get_spans_from_bio(data["labels"]):
            span = sentence[span_indices[0] : span_indices[-1] + 1]
            value = self._remap_label(label)
            if value != "O":
                span.add_label(self.label_type, value=value, score=score)

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
            sentence._has_context = True
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
        normalize_coords_to_thousands: bool = True,
        label_name_map: Optional[dict[str, str]] = None,
        **corpusargs,
    ) -> None:
        """Instantiates a Corpus from a OCR-Json format.

        :param train_path: the folder for the training data
        :param dev_path: the folder for the dev data
        :param test_path: the folder for the test data
        :param path_to_split_directory: base folder with the task data
        :param label_type: the label_type to add the ocr labels to
        :param encoding: the encoding to load the .json files with
        :param load_images: if True, the pillow images will be added as metadata
        :param normalize_coords_to_thousands: if True, the coordinates will be ranged from 0 to 1000
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param label_name_map: Optionally map tag names to different schema.
        :return: a Corpus with Sentences that contain OCR information
        """
        train: Optional[Dataset] = (
            OcrJsonDataset(
                train_path,
                label_type=label_type,
                encoding=encoding,
                in_memory=in_memory,
                load_images=load_images,
                normalize_coords_to_thousands=normalize_coords_to_thousands,
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
                normalize_coords_to_thousands=normalize_coords_to_thousands,
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
                normalize_coords_to_thousands=normalize_coords_to_thousands,
                label_name_map=label_name_map,
            )
            if test_path is not None
            else None
        )

        super().__init__(train, dev, test, **corpusargs)


class SROIE(OcrCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        encoding: str = "utf-8",
        label_type: str = "ner",
        in_memory: bool = True,
        load_images: bool = False,
        normalize_coords_to_thousands: bool = True,
        label_name_map: Optional[dict[str, str]] = None,
        **corpusargs,
    ) -> None:
        """Instantiates the SROIE corpus with perfect ocr boxes.

        :param base_path: the path to store the dataset or load it from
        :param label_type: the label_type to add the ocr labels to
        :param encoding: the encoding to load the .json files with
        :param load_images: if True, the pillow images will be added as metadata
        :param normalize_coords_to_thousands: if True, the coordinates will be ranged from 0 to 1000
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param label_name_map: Optionally map tag names to different schema.
        :return: a Corpus with Sentences that contain OCR information
        """
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

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
            normalize_coords_to_thousands=normalize_coords_to_thousands,
            **corpusargs,
        )
