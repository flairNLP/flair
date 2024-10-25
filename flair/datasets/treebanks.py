import logging
import re
from pathlib import Path
from typing import Optional, Union

import flair
from flair.data import Corpus, FlairDataset, Sentence, Token
from flair.datasets.base import find_train_dev_test_files
from flair.file_utils import cached_path

log = logging.getLogger("flair")


class UniversalDependenciesCorpus(Corpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        train_file=None,
        test_file=None,
        dev_file=None,
        in_memory: bool = True,
        split_multiwords: bool = True,
    ) -> None:
        """Instantiates a Corpus from CoNLL-U column-formatted task data such as the UD corpora.

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        :param split_multiwords: If set to True, multiwords are split (default), otherwise kept as single tokens
        :return: a Corpus with annotated train, dev and test data
        """
        # find train, dev and test files if not specified
        dev_file, test_file, train_file = find_train_dev_test_files(data_folder, dev_file, test_file, train_file)

        # get train data
        train = (
            UniversalDependenciesDataset(train_file, in_memory=in_memory, split_multiwords=split_multiwords)
            if train_file is not None
            else None
        )

        # get test data
        test = (
            UniversalDependenciesDataset(test_file, in_memory=in_memory, split_multiwords=split_multiwords)
            if test_file is not None
            else None
        )

        # get dev data
        dev = (
            UniversalDependenciesDataset(dev_file, in_memory=in_memory, split_multiwords=split_multiwords)
            if dev_file is not None
            else None
        )

        super().__init__(train, dev, test, name=str(data_folder))


class UniversalDependenciesDataset(FlairDataset):
    def __init__(
        self,
        path_to_conll_file: Union[str, Path],
        in_memory: bool = True,
        split_multiwords: bool = True,
    ) -> None:
        """Instantiates a column dataset in CoNLL-U format.

        :param path_to_conll_file: Path to the CoNLL-U formatted file
        :param in_memory: If set to True, keeps full dataset in memory, otherwise does disk reads
        """
        path_to_conll_file = Path(path_to_conll_file)
        assert path_to_conll_file.exists()

        self.in_memory: bool = in_memory
        self.split_multiwords: bool = split_multiwords

        self.path_to_conll_file = path_to_conll_file
        self.total_sentence_count: int = 0

        with open(str(self.path_to_conll_file), encoding="utf-8") as file:
            # option 1: read only sentence boundaries as offset positions
            if not self.in_memory:
                self.indices: list[int] = []

                line = file.readline()
                position = 0
                while line:
                    line = line.strip()
                    if line == "":
                        self.indices.append(position)
                        position = file.tell()
                    line = file.readline()

                self.total_sentence_count = len(self.indices)

            # option 2: keep everything in memory
            if self.in_memory:
                self.sentences: list[Sentence] = []

                while True:
                    sentence = self._read_next_sentence(file)
                    if not sentence:
                        break
                    self.sentences.append(sentence)

                self.total_sentence_count = len(self.sentences)

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self) -> int:
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        # if in memory, retrieve parsed sentence
        if self.in_memory:
            sentence = self.sentences[index]

        # else skip to position in file where sentence begins
        else:
            with open(str(self.path_to_conll_file), encoding="utf-8") as file:
                file.seek(self.indices[index])
                sentence_or_none = self._read_next_sentence(file)
                sentence = sentence_or_none if isinstance(sentence_or_none, Sentence) else Sentence("")

        return sentence

    def _read_next_sentence(self, file) -> Optional[Sentence]:
        line = file.readline()
        tokens: list[Token] = []

        # current token ID
        token_idx = 0

        # handling for the awful UD multiword format
        current_multiword_text = ""
        current_multiword_sequence = ""
        current_multiword_first_token = 0
        current_multiword_last_token = 0

        newline_reached = False
        while line:
            line = line.strip()
            fields: list[str] = re.split("\t+", line)

            # end of sentence
            if line == "":
                if len(tokens) > 0:
                    newline_reached = True
                    break

            # comments or ellipsis
            elif line.startswith("#") or "." in fields[0]:
                line = file.readline()
                continue

            # if token is a multi-word
            elif "-" in fields[0]:
                line = file.readline()

                current_multiword_first_token = int(fields[0].split("-")[0])
                current_multiword_last_token = int(fields[0].split("-")[1])
                current_multiword_text = fields[1]
                current_multiword_sequence = ""

                if self.split_multiwords:
                    continue
                else:
                    token = Token(fields[1])
                    token.add_label("lemma", str(fields[2]))
                    if len(fields) > 9 and "SpaceAfter=No" in fields[9]:
                        token.whitespace_after = 0
                    tokens.append(token)
                    token_idx += 1

            # normal single-word tokens
            else:
                # if we don't split multiwords, skip over component words
                if not self.split_multiwords and token_idx < current_multiword_last_token:
                    token_idx += 1
                    line = file.readline()
                    continue

                # add token
                token = Token(fields[1], head_id=int(fields[6]))
                token.add_label("lemma", str(fields[2]))
                token.add_label("upos", str(fields[3]))
                token.add_label("pos", str(fields[4]))
                token.add_label("dependency", str(fields[7]))

                if len(fields) > 9 and "SpaceAfter=No" in fields[9]:
                    token.whitespace_after = 0

                # add morphological tags
                for morph in str(fields[5]).split("|"):
                    if "=" not in morph:
                        continue
                    token.add_label(morph.split("=")[0].lower(), morph.split("=")[1])

                if len(fields) > 10 and str(fields[10]) == "Y":
                    token.add_label("frame", str(fields[11]))

                token_idx += 1

                # derive whitespace logic for multiwords
                if token_idx <= current_multiword_last_token:
                    current_multiword_sequence += token.text

                # if multi-word equals component tokens, there should be no whitespace
                if token_idx == current_multiword_last_token and current_multiword_sequence == current_multiword_text:
                    # go through all tokens in subword and set whitespace_after information
                    for i in range(current_multiword_last_token - current_multiword_first_token):
                        tokens[-(i + 1)].whitespace_after = 0
                tokens.append(token)

            line = file.readline()

        if newline_reached or len(tokens) > 0:
            return Sentence(tokens)
        return None


class UD_ENGLISH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{revision}"
        cached_path(f"{web_path}/en_ewt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/en_ewt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/en_ewt-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_GALICIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Galician-TreeGal/{revision}"
        cached_path(f"{web_path}/gl_treegal-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/gl_treegal-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_ANCIENT_GREEK(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-PROIEL/{revision}"
        cached_path(f"{web_path}/grc_proiel-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/grc_proiel-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/grc_proiel-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_KAZAKH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Kazakh-KTB/{revision}"
        cached_path(f"{web_path}/kk_ktb-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/kk_ktb-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_OLD_CHURCH_SLAVONIC(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Old_Church_Slavonic-PROIEL/{revision}"
        cached_path(f"{web_path}/cu_proiel-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/cu_proiel-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/cu_proiel-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_ARMENIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Armenian-ArmTDP/{revision}/"
        cached_path(f"{web_path}/hy_armtdp-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/hy_armtdp-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/hy_armtdp-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_ESTONIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Estonian-EDT/{revision}"
        cached_path(f"{web_path}/et_edt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/et_edt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/et_edt-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_GERMAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/{revision}"
        cached_path(f"{ud_path}/de_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/de_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/de_gsd-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_GERMAN_HDT(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = False,
        split_multiwords: bool = True,
        revision: str = "dev",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_German-HDT/{revision}"
        cached_path(f"{ud_path}/de_hdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/de_hdt-ud-test.conllu", Path("datasets") / dataset_name)

        train_filenames = [
            "de_hdt-ud-train-a-1.conllu",
            "de_hdt-ud-train-a-2.conllu",
            "de_hdt-ud-train-b-1.conllu",
            "de_hdt-ud-train-b-2.conllu",
        ]

        for train_file in train_filenames:
            cached_path(f"{ud_path}/{train_file}", Path("datasets") / dataset_name / "original")

        data_path = flair.cache_root / "datasets" / dataset_name

        new_train_file: Path = data_path / "de_hdt-ud-train-all.conllu"

        if not new_train_file.is_file():
            with open(new_train_file, "w") as f_out:
                for train_filename in train_filenames:
                    with open(data_path / "original" / train_filename) as f_in:
                        f_out.write(f_in.read())

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_DUTCH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Dutch-Alpino/{revision}"
        cached_path(f"{ud_path}/nl_alpino-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/nl_alpino-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/nl_alpino-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_FAROESE(UniversalDependenciesCorpus):
    """This treebank includes the Faroese treebank dataset.

    The data is obtained from the following link:
    https://github.com/UniversalDependencies/UD_Faroese-FarPaHC/tree/{revision}

    Faronese is a small Western Scandinavian language with 60.000-100.000, related to Icelandic and Old Norse.
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Faroese-FarPaHC/{revision}"
        cached_path(f"{web_path}/fo_farpahc-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/fo_farpahc-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/fo_farpahc-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_FRENCH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/{revision}"
        cached_path(f"{ud_path}/fr_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/fr_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/fr_gsd-ud-train.conllu", Path("datasets") / dataset_name)
        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_ITALIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/{revision}"
        cached_path(f"{ud_path}/it_isdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/it_isdt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/it_isdt-ud-train.conllu", Path("datasets") / dataset_name)
        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_LATIN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-LLCT/{revision}/"
        cached_path(f"{web_path}/la_llct-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/la_llct-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/la_llct-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_SPANISH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/{revision}"
        cached_path(f"{ud_path}/es_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/es_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/es_gsd-ud-train.conllu", Path("datasets") / dataset_name)
        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_PORTUGUESE(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-Bosque/{revision}"
        cached_path(f"{ud_path}/pt_bosque-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/pt_bosque-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/pt_bosque-ud-train.conllu", Path("datasets") / dataset_name)
        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_ROMANIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/{revision}"
        cached_path(f"{ud_path}/ro_rrt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ro_rrt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ro_rrt-ud-train.conllu", Path("datasets") / dataset_name)
        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_CATALAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Catalan-AnCora/{revision}"
        cached_path(f"{ud_path}/ca_ancora-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ca_ancora-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ca_ancora-ud-train.conllu", Path("datasets") / dataset_name)
        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_POLISH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-LFG/{revision}"
        cached_path(f"{ud_path}/pl_lfg-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/pl_lfg-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/pl_lfg-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_CZECH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = False,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-PDT/{revision}"
        cached_path(f"{ud_path}/cs_pdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/cs_pdt-ud-test.conllu", Path("datasets") / dataset_name)

        train_suffixes = ["ca", "ct", "la", "lt", "ma", "mt", "va"]

        for train_suffix in train_suffixes:
            cached_path(
                f"{ud_path}/cs_pdt-ud-train-{train_suffix}.conllu",
                Path("datasets") / dataset_name / "original",
            )
        data_path = flair.cache_root / "datasets" / dataset_name

        train_filenames = [f"cs_pdt-ud-train-{train_suffix}.conllu" for train_suffix in train_suffixes]

        new_train_file: Path = data_path / "cs_pdt-ud-train-all.conllu"

        if not new_train_file.is_file():
            with open(new_train_file, "w") as f_out:
                for train_filename in train_filenames:
                    with open(data_path / "original" / train_filename) as f_in:
                        f_out.write(f_in.read())
        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_SLOVAK(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Slovak-SNK/{revision}"
        cached_path(f"{ud_path}/sk_snk-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sk_snk-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sk_snk-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_SWEDISH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/{revision}"
        cached_path(f"{ud_path}/sv_talbanken-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sv_talbanken-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sv_talbanken-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_DANISH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Danish-DDT/{revision}"
        cached_path(f"{ud_path}/da_ddt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/da_ddt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/da_ddt-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_NORWEGIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Norwegian-Bokmaal/{revision}"
        cached_path(f"{ud_path}/no_bokmaal-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/no_bokmaal-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/no_bokmaal-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_FINNISH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Finnish-TDT/{revision}"
        cached_path(f"{ud_path}/fi_tdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/fi_tdt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/fi_tdt-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_SLOVENIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Slovenian-SSJ/{revision}"
        cached_path(f"{ud_path}/sl_ssj-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sl_ssj-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sl_ssj-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_CROATIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Croatian-SET/{revision}"
        cached_path(f"{ud_path}/hr_set-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/hr_set-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/hr_set-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_SERBIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Serbian-SET/{revision}"
        cached_path(f"{ud_path}/sr_set-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sr_set-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/sr_set-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_BULGARIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Bulgarian-BTB/{revision}"
        cached_path(f"{ud_path}/bg_btb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/bg_btb-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/bg_btb-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_ARABIC(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/{revision}"
        cached_path(f"{ud_path}/ar_padt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ar_padt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ar_padt-ud-train.conllu", Path("datasets") / dataset_name)
        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_HEBREW(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/{revision}"
        cached_path(f"{ud_path}/he_htb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/he_htb-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/he_htb-ud-train.conllu", Path("datasets") / dataset_name)
        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_TURKISH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/{revision}"
        cached_path(f"{ud_path}/tr_imst-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/tr_imst-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/tr_imst-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_UKRAINIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-IU/{revision}"
        cached_path(f"{ud_path}/uk_iu-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/uk_iu-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/uk_iu-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_PERSIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Persian-Seraji/{revision}"
        cached_path(f"{ud_path}/fa_seraji-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/fa_seraji-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/fa_seraji-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_RUSSIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/{revision}"
        cached_path(f"{ud_path}/ru_syntagrus-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ru_syntagrus-ud-test.conllu", Path("datasets") / dataset_name)

        train_filenames = [
            "ru_syntagrus-ud-train-a.conllu",
            "ru_syntagrus-ud-train-b.conllu",
            "ru_syntagrus-ud-train-c.conllu",
        ]

        for train_file in train_filenames:
            cached_path(f"{ud_path}/{train_file}", Path("datasets") / dataset_name / "original")

        data_path = flair.cache_root / "datasets" / dataset_name

        new_train_file: Path = data_path / "ru_syntagrus-ud-train-all.conllu"

        if not new_train_file.is_file():
            with open(new_train_file, "w") as f_out:
                for train_filename in train_filenames:
                    with open(data_path / "original" / train_filename) as f_in:
                        f_out.write(f_in.read())

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_HINDI(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Hindi-HDTB/{revision}"
        cached_path(f"{ud_path}/hi_hdtb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/hi_hdtb-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/hi_hdtb-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_INDONESIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-GSD/{revision}"
        cached_path(f"{ud_path}/id_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/id_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/id_gsd-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_JAPANESE(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/{revision}"
        cached_path(f"{ud_path}/ja_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ja_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ja_gsd-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_CHINESE(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/{revision}"
        cached_path(f"{ud_path}/zh_gsd-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/zh_gsd-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/zh_gsd-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_KOREAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Korean-Kaist/{revision}"
        cached_path(f"{ud_path}/ko_kaist-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ko_kaist-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/ko_kaist-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_BASQUE(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Basque-BDT/{revision}"
        cached_path(f"{ud_path}/eu_bdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/eu_bdt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/eu_bdt-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_CHINESE_KYOTO(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/{revision}"
        cached_path(f"{web_path}/lzh_kyoto-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/lzh_kyoto-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/lzh_kyoto-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_GREEK(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GDT/{revision}"
        cached_path(f"{web_path}/el_gdt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/el_gdt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/el_gdt-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_NAIJA(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Naija-NSC/{revision}"
        cached_path(f"{web_path}//pcm_nsc-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}//pcm_nsc-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}//pcm_nsc-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_LIVVI(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Livvi-KKPP/{revision}"
        cached_path(f"{web_path}/olo_kkpp-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/olo_kkpp-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_BURYAT(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Buryat-BDT/{revision}"
        cached_path(f"{web_path}/bxr_bdt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/bxr_bdt-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_NORTH_SAMI(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_North_Sami-Giella/{revision}"
        cached_path(f"{web_path}/sme_giella-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/sme_giella-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_MARATHI(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Marathi-UFAL/{revision}"
        cached_path(f"{web_path}/mr_ufal-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/mr_ufal-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/mr_ufal-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_MALTESE(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Maltese-MUDT/{revision}"
        cached_path(f"{web_path}/mt_mudt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/mt_mudt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/mt_mudt-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_AFRIKAANS(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Afrikaans-AfriBooms/{revision}"
        cached_path(f"{web_path}/af_afribooms-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/af_afribooms-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/af_afribooms-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_GOTHIC(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Gothic-PROIEL/{revision}"
        cached_path(f"{web_path}/got_proiel-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/got_proiel-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/got_proiel-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_OLD_FRENCH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Old_French-SRCMF/{revision}"

        cached_path(f"{web_path}/fro_profiterole-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/fro_profiterole-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/fro_profiterole-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_WOLOF(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Wolof-WTB/{revision}"
        cached_path(f"{web_path}/wo_wtb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/wo_wtb-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/wo_wtb-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_BELARUSIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Belarusian-HSE/{revision}"
        cached_path(f"{web_path}/be_hse-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/be_hse-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/be_hse-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_COPTIC(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Coptic-Scriptorium/{revision}"
        cached_path(f"{web_path}/cop_scriptorium-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(
            f"{web_path}/cop_scriptorium-ud-test.conllu",
            Path("datasets") / dataset_name,
        )
        cached_path(
            f"{web_path}/cop_scriptorium-ud-train.conllu",
            Path("datasets") / dataset_name,
        )

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_IRISH(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Irish-IDT/{revision}"
        cached_path(f"{web_path}/ga_idt-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/ga_idt-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/ga_idt-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_LATVIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/{revision}"
        cached_path(f"{web_path}/lv_lvtb-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/lv_lvtb-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/lv_lvtb-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_LITHUANIAN(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "master",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Lithuanian-ALKSNIS/{revision}"
        cached_path(f"{web_path}/lt_alksnis-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/lt_alksnis-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{web_path}/lt_alksnis-ud-train.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


class UD_BAVARIAN_MAIBAAM(UniversalDependenciesCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        split_multiwords: bool = True,
        revision: str = "dev",
    ) -> None:
        base_path = Path(flair.cache_root) / "datasets" if not base_path else Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root

        data_folder = base_path / dataset_name

        # download data if necessary
        web_path = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Bavarian-MaiBaam/{revision}"
        cached_path(f"{web_path}/bar_maibaam-ud-test.conllu", Path("datasets") / dataset_name)

        super().__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)
