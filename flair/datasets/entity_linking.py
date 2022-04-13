import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests

import flair
from flair.data import Corpus, MultiCorpus, Sentence
from flair.datasets.sequence_labeling import ColumnCorpus
from flair.file_utils import cached_path, unpack_file
from flair.tokenization import SegtokSentenceSplitter, SentenceSplitter

log = logging.getLogger("flair")


class NEL_ENGLISH_AQUAINT(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        agreement_threshold: float = 0.5,
        sentence_splitter: SentenceSplitter = SegtokSentenceSplitter(),
        **corpusargs,
    ):
        """
        Initialize Aquaint Entity Linking corpus introduced in: D. Milne and I. H. Witten.
        Learning to link with wikipedia
        (https://www.cms.waikato.ac.nz/~ihw/papers/08-DNM-IHW-LearningToLinkWithWikipedia.pdf).
        If you call the constructor the first time the dataset gets automatically downloaded and transformed in
        tab-separated column format (aquaint.txt).

        Parameters
        ----------
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        in_memory: If True, keeps dataset in memory giving speedups in training.
        agreement_threshold: Some link annotations come with an agreement_score representing the agreement from the human annotators. The score ranges from lowest 0.2
                             to highest 1.0. The lower the score, the less "important" is the entity because fewer annotators thought it was worth linking.
                             Default is 0.5 which means the majority of annotators must have annoteted the respective entity mention.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        self.agreement_threshold = agreement_threshold

        # this dataset name
        dataset_name = self.__class__.__name__.lower() + "_" + type(sentence_splitter).__name__

        # default dataset folder is the cache root

        data_folder = base_path / dataset_name

        aquaint_el_path = "https://www.nzdl.org/wikification/data/wikifiedStories.zip"
        corpus_file_name = "aquaint.txt"
        parsed_dataset = data_folder / corpus_file_name

        # download and parse data if necessary
        if not parsed_dataset.exists():
            aquaint_el_zip = cached_path(f"{aquaint_el_path}", Path("datasets") / dataset_name)
            unpack_file(aquaint_el_zip, data_folder, "zip", False)

            try:
                with open(parsed_dataset, "w", encoding="utf-8") as txt_out:

                    # iterate over all html files
                    for file in os.listdir(data_folder):

                        if not file.endswith(".htm"):
                            continue

                        with open(str(data_folder / file), "r", encoding="utf-8") as txt_in:
                            text = txt_in.read()

                        # get rid of html syntax, we only need the text
                        strings = text.split("<p> ")
                        strings[0] = strings[0].split('<h1 id="header">')[1][:-7]

                        for i in range(1, len(strings) - 1):
                            strings[i] = strings[i][:-7]

                        strings[-1] = strings[-1][:-23]

                        # between all documents we write a separator symbol
                        txt_out.write("-DOCSTART-\n\n")

                        for string in strings:

                            # skip empty strings
                            if not string:
                                continue

                            # process the annotation format in the text and collect triples (begin_mention, length_mention, wikiname)
                            indices = []
                            lengths = []
                            wikinames = []

                            current_entity = string.find("[[")  # each annotation starts with '[['
                            while current_entity != -1:
                                wikiname = ""
                                surface_form = ""
                                j = current_entity + 2

                                while string[j] not in ["]", "|"]:
                                    wikiname += string[j]
                                    j += 1

                                if string[j] == "]":  # entity mention ends, i.e. looks like this [[wikiname]]
                                    surface_form = wikiname  # in this case entity mention = wiki-page name
                                else:  # string[j] == '|'
                                    j += 1
                                    while string[j] not in ["]", "|"]:
                                        surface_form += string[j]
                                        j += 1

                                    if (
                                        string[j] == "|"
                                    ):  # entity has a score, i.e. looks like this [[wikiname|surface_form|agreement_score]]
                                        agreement_score = float(string[j + 1 : j + 4])
                                        j += 4  # points to first ']' of entity now
                                        if agreement_score < self.agreement_threshold:  # discard entity
                                            string = string[:current_entity] + surface_form + string[j + 2 :]
                                            current_entity = string.find("[[")
                                            continue

                                # replace [[wikiname|surface_form|score]] by surface_form and save index, length and wikiname of mention
                                indices.append(current_entity)
                                lengths.append(len(surface_form))
                                wikinames.append(wikiname[0].upper() + wikiname.replace(" ", "_")[1:])

                                string = string[:current_entity] + surface_form + string[j + 2 :]

                                current_entity = string.find("[[")

                            # sentence splitting and tokenization
                            sentences = sentence_splitter.split(string)
                            sentence_offsets = [sentence.start_pos or 0 for sentence in sentences]

                            # iterate through all annotations and add to corresponding tokens
                            for mention_start, mention_length, wikiname in zip(indices, lengths, wikinames):

                                # find sentence to which annotation belongs
                                sentence_index = 0
                                for i in range(1, len(sentences)):
                                    if mention_start < sentence_offsets[i]:
                                        break
                                    else:
                                        sentence_index += 1

                                # position within corresponding sentence
                                mention_start -= sentence_offsets[sentence_index]
                                mention_end = mention_start + mention_length

                                # set annotation for tokens of entity mention
                                first = True
                                for token in sentences[sentence_index].tokens:
                                    assert token.start_pos is not None
                                    assert token.end_pos is not None
                                    if (
                                        token.start_pos >= mention_start and token.end_pos <= mention_end
                                    ):  # token belongs to entity mention
                                        if first:
                                            token.set_label(typename="nel", value="B-" + wikiname)
                                            first = False
                                        else:
                                            token.set_label(typename="nel", value="I-" + wikiname)

                            # write to out-file in column format
                            for sentence in sentences:

                                for token in sentence.tokens:

                                    labels = token.get_labels("nel")

                                    if len(labels) == 0:  # no entity
                                        txt_out.write(token.text + "\tO\n")

                                    else:  # annotation
                                        txt_out.write(token.text + "\t" + labels[0].value + "\n")

                                txt_out.write("\n")  # empty line after each sentence

            except Exception:
                # in case something goes wrong, delete the dataset and raise error
                os.remove(parsed_dataset)
                raise

        super(NEL_ENGLISH_AQUAINT, self).__init__(
            data_folder,
            column_format={0: "text", 1: "nel"},
            train_file=corpus_file_name,
            in_memory=in_memory,
            **corpusargs,
        )


class NEL_GERMAN_HIPE(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        wiki_language: str = "dewiki",
        **corpusargs,
    ):
        """
        Initialize a sentence-segmented version of the HIPE entity linking corpus for historical German (see description
        of HIPE at https://impresso.github.io/CLEF-HIPE-2020/). This version was segmented by @stefan-it and is hosted
        at https://github.com/stefan-it/clef-hipe.
        If you call the constructor the first time the dataset gets automatically downloaded and transformed in
        tab-separated column format.

        Parameters
        ----------
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        in_memory: If True, keeps dataset in memory giving speedups in training.
        wiki_language : specify the language of the names of the wikipedia pages, i.e. which language version of
        Wikipedia URLs to use. Since the text is in german the default language is German.
        """
        self.wiki_language = wiki_language
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        dev_raw_url = "https://raw.githubusercontent.com/stefan-it/clef-hipe/main/data/future/dev-v1.2/de/HIPE-data-v1.2-dev-de-normalized-manual-eos.tsv"
        test_raw_url = "https://raw.githubusercontent.com/stefan-it/clef-hipe/main/data/future/test-v1.3/de/HIPE-data-v1.3-test-de-normalized-manual-eos.tsv"
        train_raw_url = "https://raw.githubusercontent.com/stefan-it/clef-hipe/main/data/future/training-v1.2/de/HIPE-data-v1.2-train-de-normalized-manual-eos.tsv"
        train_file_name = wiki_language + "_train.tsv"
        parsed_dataset = data_folder / train_file_name

        # download and parse data if necessary
        if not parsed_dataset.exists():

            # from qwikidata.linked_data_interface import get_entity_dict_from_api

            original_train_path = cached_path(f"{train_raw_url}", Path("datasets") / dataset_name)
            original_test_path = cached_path(f"{test_raw_url}", Path("datasets") / dataset_name)
            original_dev_path = cached_path(f"{dev_raw_url}", Path("datasets") / dataset_name)

            # generate qid wikiname dictionaries
            log.info("Get wikinames from wikidata...")
            train_dict = self._get_qid_wikiname_dict(path=original_train_path)
            test_dict = self._get_qid_wikiname_dict(original_test_path)
            dev_dict = self._get_qid_wikiname_dict(original_dev_path)
            log.info("...done!")

            # merge dictionaries
            qid_wikiname_dict = {**train_dict, **test_dict, **dev_dict}

            for doc_path, file_name in zip(
                [original_train_path, original_test_path, original_dev_path],
                [
                    train_file_name,
                    wiki_language + "_test.tsv",
                    wiki_language + "_dev.tsv",
                ],
            ):
                with open(doc_path, "r", encoding="utf-8") as read, open(
                    data_folder / file_name, "w", encoding="utf-8"
                ) as write:

                    # ignore first line
                    read.readline()
                    line = read.readline()
                    last_eos = True

                    while line:
                        # commented and empty lines
                        if line[0] == "#" or line == "\n":
                            if line[2:13] == "document_id":  # beginning of new document

                                if last_eos:
                                    write.write("-DOCSTART-\n\n")
                                    last_eos = False
                                else:
                                    write.write("\n-DOCSTART-\n\n")

                        else:
                            line_list = line.split("\t")
                            if not line_list[7] in [
                                "_",
                                "NIL",
                            ]:  # line has wikidata link

                                wikiname = qid_wikiname_dict[line_list[7]]

                                if wikiname != "O":
                                    annotation = line_list[1][:2] + wikiname
                                else:  # no entry in chosen language
                                    annotation = "O"

                            else:

                                annotation = "O"

                            write.write(line_list[0] + "\t" + annotation + "\n")

                            if line_list[-1][-4:-1] == "EOS":  # end of sentence
                                write.write("\n")
                                last_eos = True
                            else:
                                last_eos = False

                        line = read.readline()

        super(NEL_GERMAN_HIPE, self).__init__(
            data_folder,
            column_format={0: "text", 1: "nel"},
            train_file=train_file_name,
            dev_file=wiki_language + "_dev.tsv",
            test_file=wiki_language + "_test.tsv",
            in_memory=in_memory,
            **corpusargs,
        )

    def _get_qid_wikiname_dict(self, path):

        qid_set = set()
        with open(path, mode="r", encoding="utf-8") as read:
            # read all Q-IDs

            # ignore first line
            read.readline()
            line = read.readline()

            while line:

                if not (line[0] == "#" or line == "\n"):  # commented or empty lines
                    line_list = line.split("\t")
                    if not line_list[7] in ["_", "NIL"]:  # line has wikidata link

                        qid_set.add(line_list[7])

                line = read.readline()

        base_url = (
            "https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=sitelinks&sitefilter="
            + self.wiki_language
            + "&ids="
        )

        qid_list = list(qid_set)
        ids = ""
        length = len(qid_list)
        qid_wikiname_dict = {}
        for i in range(length):
            if (
                i + 1
            ) % 50 == 0 or i == length - 1:  # there is a limit to the number of ids in one request in the wikidata api

                ids += qid_list[i]
                # request
                response_json = requests.get(base_url + ids).json()

                for qid in response_json["entities"]:

                    try:
                        wikiname = response_json["entities"][qid]["sitelinks"][self.wiki_language]["title"].replace(
                            " ", "_"
                        )
                    except KeyError:  # language not available for specific wikiitem
                        wikiname = "O"

                    qid_wikiname_dict[qid] = wikiname

                ids = ""

            else:
                ids += qid_list[i]
                ids += "|"

        return qid_wikiname_dict


class NEL_ENGLISH_AIDA(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        check_existence: bool = False,
        **corpusargs,
    ):
        """
        Initialize AIDA CoNLL-YAGO Entity Linking corpus introduced here https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads.
        License: https://creativecommons.org/licenses/by-sa/3.0/deed.en_US
        If you call the constructor the first time the dataset gets automatically downloaded and transformed in tab-separated column format.

        Parameters
        ----------
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        in_memory: If True, keeps dataset in memory giving speedups in training.
        check_existence: If True the existence of the given wikipedia ids/pagenames is checked and non existent ids/names will be igrnored.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        conll_yago_path = "https://nlp.informatik.hu-berlin.de/resources/datasets/conll_entity_linking/"
        corpus_file_name = "train"
        parsed_dataset = data_folder / corpus_file_name

        if not parsed_dataset.exists():

            import wikipediaapi

            wiki_wiki = wikipediaapi.Wikipedia(language="en")

            testa_unprocessed_path = cached_path(f"{conll_yago_path}aida_conll_testa", Path("datasets") / dataset_name)
            testb_unprocessed_path = cached_path(f"{conll_yago_path}aida_conll_testb", Path("datasets") / dataset_name)
            train_unprocessed_path = cached_path(f"{conll_yago_path}aida_conll_train", Path("datasets") / dataset_name)

            # we use the wikiids in the data instead of directly utilizing the wikipedia urls.
            # like this we can quickly check if the corresponding page exists
            wikiid_wikiname_dict = self._get_wikiid_wikiname_dict(data_folder)

            for name, path in zip(
                ["train", "testa", "testb"],
                [
                    train_unprocessed_path,
                    testa_unprocessed_path,
                    testb_unprocessed_path,
                ],
            ):
                with open(data_folder / name, "w", encoding="utf-8") as write, open(
                    path, "r", encoding="utf-8"
                ) as read:

                    for line in read:

                        line_list = line.split("\t")
                        if len(line_list) <= 4:
                            if line_list[0][:10] == "-DOCSTART-":  # Docstart
                                write.write("-DOCSTART-\n\n")
                            elif line_list[0] == "\n":  # empty line
                                write.write("\n")
                            else:  # text without annotation or marked '--NME--' (no matching entity)
                                if len(line_list) == 1:
                                    write.write(line_list[0][:-1] + "\tO\n")
                                else:
                                    write.write(line_list[0] + "\tO\n")
                        else:  # line with annotation
                            wikiname = wikiid_wikiname_dict[line_list[5].strip()]
                            if wikiname != "O":
                                write.write(line_list[0] + "\t" + line_list[1] + "-" + wikiname + "\n")
                            else:
                                # if there is a bad wikiid we can check if the given url in the data exists using wikipediaapi
                                wikiname = line_list[4].split("/")[-1]
                                if check_existence:
                                    page = wiki_wiki.page(wikiname)
                                    if page.exists():
                                        write.write(line_list[0] + "\t" + line_list[1] + "-" + wikiname + "\n")
                                    else:  # neither the wikiid nor the url exist
                                        write.write(line_list[0] + "\tO\n")
                                else:
                                    write.write(line_list[0] + "\t" + line_list[1] + "-" + wikiname + "\n")

                # delete unprocessed file
                os.remove(path)

        super(NEL_ENGLISH_AIDA, self).__init__(
            data_folder,
            column_format={0: "text", 1: "nel"},
            train_file=corpus_file_name,
            dev_file="testa",
            test_file="testb",
            in_memory=in_memory,
            **corpusargs,
        )

    def _get_wikiid_wikiname_dict(self, base_folder):

        # collect all wikiids
        wikiid_set = set()
        for data_file in ["aida_conll_testa", "aida_conll_testb", "aida_conll_train"]:
            with open(base_folder / data_file, mode="r", encoding="utf-8") as read:
                line = read.readline()
                while line:
                    row = line.split("\t")
                    if len(row) > 4:  # line has a wiki annotation
                        wikiid_set.add(row[5].strip())
                    line = read.readline()

        # create the dictionary
        wikiid_wikiname_dict = {}
        wikiid_list = list(wikiid_set)
        ids = ""
        length = len(wikiid_list)

        for i in range(length):
            if (
                i + 1
            ) % 50 == 0 or i == length - 1:  # there is a limit to the number of ids in one request in the wikimedia api

                ids += wikiid_list[i]
                # request
                resp = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "info",
                        "pageids": ids,
                        "format": "json",
                    },
                ).json()

                for wikiid in resp["query"]["pages"]:
                    try:
                        wikiname = resp["query"]["pages"][wikiid]["title"].replace(" ", "_")
                    except KeyError:  # bad wikiid
                        wikiname = "O"
                    wikiid_wikiname_dict[wikiid] = wikiname
                ids = ""

            else:
                ids += wikiid_list[i]
                ids += "|"

        return wikiid_wikiname_dict


class NEL_ENGLISH_IITB(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        ignore_disagreements: bool = False,
        sentence_splitter: SentenceSplitter = SegtokSentenceSplitter(),
        **corpusargs,
    ):
        """
        Initialize ITTB Entity Linking corpus introduced in "Collective Annotation of Wikipedia Entities in Web Text" Sayali Kulkarni, Amit Singh, Ganesh Ramakrishnan, and Soumen Chakrabarti.
        If you call the constructor the first time the dataset gets automatically downloaded and transformed in tab-separated column format.

        Parameters
        ----------
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        in_memory: If True, keeps dataset in memory giving speedups in training.
        ignore_disagreements: If True annotations with annotator disagreement will be ignored.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower() + "_" + type(sentence_splitter).__name__

        data_folder = base_path / dataset_name

        iitb_el_docs_path = "https://www.cse.iitb.ac.in/~soumen/doc/CSAW/Annot/CSAW_crawledDocs.tar.gz"
        iitb_el_annotations_path = "https://www.cse.iitb.ac.in/~soumen/doc/CSAW/Annot/CSAW_Annotations.xml"
        corpus_file_name = "iitb.txt"
        parsed_dataset = data_folder / corpus_file_name

        if not parsed_dataset.exists():

            docs_zip_path = cached_path(f"{iitb_el_docs_path}", Path("datasets") / dataset_name)
            annotations_xml_path = cached_path(f"{iitb_el_annotations_path}", Path("datasets") / dataset_name)

            unpack_file(docs_zip_path, data_folder, "tar", False)

            import xml.etree.ElementTree as ET

            tree = ET.parse(annotations_xml_path)
            root = tree.getroot()

            # names of raw text documents
            doc_names = set()
            for elem in root:
                if elem[0].text is not None:
                    doc_names.add(elem[0].text)

            # open output_file
            with open(parsed_dataset, "w", encoding="utf-8") as write:
                # iterate through all documents
                for doc_name in doc_names:
                    with open(data_folder / "crawledDocs" / doc_name, "r", encoding="utf-8") as read:
                        text = read.read()

                        # split sentences and tokenize
                        sentences = sentence_splitter.split(text)
                        sentence_offsets = [sentence.start_pos or 0 for sentence in sentences]

                        # iterate through all annotations and add to corresponding tokens
                        for elem in root:

                            if elem[0].text == doc_name and elem[2].text:  # annotation belongs to current document

                                wikiname = elem[2].text.replace(" ", "_")
                                assert elem[3].text is not None
                                assert elem[4].text is not None
                                mention_start = int(elem[3].text)
                                mention_length = int(elem[4].text)

                                # find sentence to which annotation belongs
                                sentence_index = 0
                                for i in range(1, len(sentences)):
                                    if mention_start < sentence_offsets[i]:
                                        break
                                    else:
                                        sentence_index += 1

                                # position within corresponding sentence
                                mention_start -= sentence_offsets[sentence_index]
                                mention_end = mention_start + mention_length

                                # set annotation for tokens of entity mention
                                first = True
                                for token in sentences[sentence_index].tokens:
                                    assert token.start_pos is not None
                                    assert token.end_pos is not None
                                    if (
                                        token.start_pos >= mention_start and token.end_pos <= mention_end
                                    ):  # token belongs to entity mention
                                        assert elem[1].text is not None
                                        if first:
                                            token.set_label(
                                                typename=elem[1].text,
                                                value="B-" + wikiname,
                                            )
                                            first = False
                                        else:
                                            token.set_label(
                                                typename=elem[1].text,
                                                value="I-" + wikiname,
                                            )

                        # write to out file
                        write.write("-DOCSTART-\n\n")  # each file is one document

                        for sentence in sentences:

                            for token in sentence.tokens:

                                labels = token.labels

                                if len(labels) == 0:  # no entity
                                    write.write(token.text + "\tO\n")

                                elif len(labels) == 1:  # annotation from one annotator
                                    write.write(token.text + "\t" + labels[0].value + "\n")

                                else:  # annotations from two annotators

                                    if labels[0].value == labels[1].value:  # annotators agree
                                        write.write(token.text + "\t" + labels[0].value + "\n")

                                    else:  # annotators disagree: ignore or arbitrarily take first annotation

                                        if ignore_disagreements:
                                            write.write(token.text + "\tO\n")

                                        else:
                                            write.write(token.text + "\t" + labels[0].value + "\n")

                            write.write("\n")  # empty line after each sentence

        super(NEL_ENGLISH_IITB, self).__init__(
            data_folder,
            column_format={0: "text", 1: "nel"},
            train_file=corpus_file_name,
            in_memory=in_memory,
            **corpusargs,
        )


class NEL_ENGLISH_TWEEKI(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize Tweeki Entity Linking corpus introduced in "Tweeki: Linking Named Entities on Twitter to a Knowledge Graph" Harandizadeh, Singh.
        The data consits of tweets with manually annotated wikipedia links.
        If you call the constructor the first time the dataset gets automatically downloaded and transformed in tab-separated column format.

        Parameters
        ----------
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        tweeki_gold_el_path = "https://raw.githubusercontent.com/ucinlp/tweeki/main/data/Tweeki_gold/Tweeki_gold"
        corpus_file_name = "tweeki_gold.txt"
        parsed_dataset = data_folder / corpus_file_name

        # download and parse data if necessary
        if not parsed_dataset.exists():

            original_file_path = cached_path(f"{tweeki_gold_el_path}", Path("datasets") / dataset_name)

            with open(original_file_path, "r", encoding="utf-8") as read, open(
                parsed_dataset, "w", encoding="utf-8"
            ) as write:
                line = read.readline()
                while line:
                    if line.startswith("#"):
                        out_line = ""
                    elif line == "\n":  # tweet ends
                        out_line = "\n-DOCSTART-\n\n"
                    else:
                        line_list = line.split("\t")
                        out_line = line_list[1] + "\t"
                        if line_list[3] == "-\n":  # no wiki name
                            out_line += "O\n"
                        else:
                            out_line += line_list[2][:2] + line_list[3].split("|")[0].replace(" ", "_") + "\n"
                    write.write(out_line)
                    line = read.readline()

            os.rename(original_file_path, str(original_file_path) + "_original")

        super(NEL_ENGLISH_TWEEKI, self).__init__(
            data_folder,
            column_format={0: "text", 1: "nel"},
            train_file=corpus_file_name,
            in_memory=in_memory,
            **corpusargs,
        )


class NEL_ENGLISH_REDDIT(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        **corpusargs,
    ):
        """
        Initialize the Reddit Entity Linking corpus containing gold annotations only (https://arxiv.org/abs/2101.01228v2) in the NER-like column format.
        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        # download and parse data if necessary
        reddit_el_path = "https://zenodo.org/record/3970806/files/reddit_el.zip"
        corpus_file_name = "reddit_el_gold.txt"
        parsed_dataset = data_folder / corpus_file_name

        if not parsed_dataset.exists():
            reddit_el_zip = cached_path(f"{reddit_el_path}", Path("datasets") / dataset_name)
            unpack_file(reddit_el_zip, data_folder, "zip", False)

            with open(data_folder / corpus_file_name, "w", encoding="utf-8") as txtout:

                # First parse the post titles
                with open(data_folder / "posts.tsv", "r", encoding="utf-8") as tsvin1, open(
                    data_folder / "gold_post_annotations.tsv", "r", encoding="utf-8"
                ) as tsvin2:

                    posts = csv.reader(tsvin1, delimiter="\t")
                    self.post_annotations = csv.reader(tsvin2, delimiter="\t")
                    self.curr_annot = next(self.post_annotations)

                    for row in posts:  # Go through all the post titles

                        txtout.writelines("-DOCSTART-\n\n")  # Start each post with a -DOCSTART- token

                        # Keep track of how many and which entity mentions does a given post title have
                        link_annots = []  # [start pos, end pos, wiki page title] of an entity mention

                        # Check if the current post title has an entity link and parse accordingly
                        if row[0] == self.curr_annot[0]:

                            link_annots.append(
                                (
                                    int(self.curr_annot[4]),
                                    int(self.curr_annot[5]),
                                    self.curr_annot[3],
                                )
                            )
                            link_annots = self._fill_annot_array(link_annots, row[0], post_flag=True)

                            # Post titles with entity mentions (if any) are handled via this function
                            self._text_to_cols(
                                Sentence(row[2], use_tokenizer=True),
                                link_annots,
                                txtout,
                            )
                        else:
                            self._text_to_cols(
                                Sentence(row[2], use_tokenizer=True),
                                link_annots,
                                txtout,
                            )

                # Then parse the comments
                with open(data_folder / "comments.tsv", "r", encoding="utf-8") as tsvin3, open(
                    data_folder / "gold_comment_annotations.tsv", "r", encoding="utf-8"
                ) as tsvin4:

                    self.comments = csv.reader(tsvin3, delimiter="\t")
                    self.comment_annotations = csv.reader(tsvin4, delimiter="\t")
                    self.curr_annot = next(self.comment_annotations)
                    self.curr_row: Optional[List[str]] = next(self.comments)
                    self.stop_iter = False

                    # Iterate over the comments.tsv file, until the end is reached
                    while not self.stop_iter:

                        txtout.writelines("-DOCSTART-\n")  # Start each comment thread with a -DOCSTART- token

                        # Keep track of the current comment thread and its corresponding key, on which the annotations are matched.
                        # Each comment thread is handled as one 'document'.
                        self.curr_comm: str = self.curr_row[4]
                        comm_key = self.curr_row[0]

                        # Python's csv package for some reason fails to correctly parse a handful of rows inside the comments.tsv file.
                        # This if-condition is needed to handle this problem.
                        if comm_key in {"en5rf4c", "es3ia8j", "es3lrmw"}:
                            if comm_key == "en5rf4c":
                                self.parsed_row = (r.split("\t") for r in self.curr_row[4].split("\n"))
                                self.curr_comm = next(self.parsed_row)  # type: ignore
                            self._fill_curr_comment(fix_flag=True)
                        # In case we are dealing with properly parsed rows, proceed with a regular parsing procedure
                        else:
                            self._fill_curr_comment(fix_flag=False)

                        link_annots = []  # [start pos, end pos, wiki page title] of an entity mention

                        # Check if the current comment thread has an entity link and parse accordingly, same as with post titles above
                        if comm_key == self.curr_annot[0]:
                            link_annots.append(
                                (
                                    int(self.curr_annot[4]),
                                    int(self.curr_annot[5]),
                                    self.curr_annot[3],
                                )
                            )
                            link_annots = self._fill_annot_array(link_annots, comm_key, post_flag=False)
                            self._text_to_cols(
                                Sentence(self.curr_comm, use_tokenizer=True),
                                link_annots,
                                txtout,
                            )
                        else:
                            # In two of the comment thread a case of capital letter spacing occurs, which the SegtokTokenizer cannot properly handle.
                            # The following if-elif condition handles these two cases and as result writes full capitalized words in each corresponding row,
                            # and not just single letters into single rows.
                            if comm_key == "dv74ybb":
                                self.curr_comm = " ".join(
                                    [word.replace(" ", "") for word in self.curr_comm.split("  ")]
                                )
                            elif comm_key == "eci2lut":
                                self.curr_comm = (
                                    self.curr_comm[:18]
                                    + self.curr_comm[18:27].replace(" ", "")
                                    + self.curr_comm[27:55]
                                    + self.curr_comm[55:68].replace(" ", "")
                                    + self.curr_comm[68:85]
                                    + self.curr_comm[85:92].replace(" ", "")
                                    + self.curr_comm[92:]
                                )

                            self._text_to_cols(
                                Sentence(self.curr_comm, use_tokenizer=True),
                                link_annots,
                                txtout,
                            )

        super(NEL_ENGLISH_REDDIT, self).__init__(
            data_folder,
            column_format={0: "text", 1: "nel"},
            train_file=corpus_file_name,
            in_memory=in_memory,
            **corpusargs,
        )

    def _text_to_cols(self, sentence: Sentence, links: list, outfile):
        """
        Convert a tokenized sentence into column format
        :param sentence: Flair Sentence object containing a tokenized post title or comment thread
        :param links: array containing information about the starting and ending position of an entity mention, as well
        as its corresponding wiki tag
        :param outfile: file, to which the output is written
        """
        for i in range(0, len(sentence)):
            # If there are annotated entity mentions for given post title or a comment thread
            if links:
                # Keep track which is the correct corresponding entity link, in cases where there is >1 link in a sentence
                link_index = [
                    j for j, v in enumerate(links) if (sentence[i].start_pos >= v[0] and sentence[i].end_pos <= v[1])
                ]
                # Write the token with a corresponding tag to file
                try:
                    if any(sentence[i].start_pos == v[0] and sentence[i].end_pos == v[1] for j, v in enumerate(links)):
                        outfile.writelines(sentence[i].text + "\tS-" + links[link_index[0]][2] + "\n")
                    elif any(
                        sentence[i].start_pos == v[0] and sentence[i].end_pos != v[1] for j, v in enumerate(links)
                    ):
                        outfile.writelines(sentence[i].text + "\tB-" + links[link_index[0]][2] + "\n")
                    elif any(
                        sentence[i].start_pos >= v[0] and sentence[i].end_pos <= v[1] for j, v in enumerate(links)
                    ):
                        outfile.writelines(sentence[i].text + "\tI-" + links[link_index[0]][2] + "\n")
                    else:
                        outfile.writelines(sentence[i].text + "\tO\n")
                # IndexError is raised in cases when there is exactly one link in a sentence, therefore can be dismissed
                except IndexError:
                    pass

            # If a comment thread or a post title has no entity link, all tokens are assigned the O tag
            else:
                outfile.writelines(sentence[i].text + "\tO\n")

            # Prevent writing empty lines if e.g. a quote comes after a dot or initials are tokenized
            # incorrectly, in order to keep the desired format (empty line as a sentence separator).
            try:
                if (
                    (sentence[i].text in {".", "!", "?", "!*"})
                    and (sentence[i + 1].text not in {'"', "“", "'", "''", "!", "?", ";)", "."})
                    and ("." not in sentence[i - 1].text)
                ):
                    outfile.writelines("\n")
            except IndexError:
                # Thrown when the second check above happens, but the last token of a sentence is reached.
                # Indicates that the EOS punctuaion mark is present, therefore an empty line needs to be written below.
                outfile.writelines("\n")

        # If there is no punctuation mark indicating EOS, an empty line is still needed after the EOS
        if sentence[-1].text not in {".", "!", "?"}:
            outfile.writelines("\n")

    def _fill_annot_array(self, annot_array: list, key: str, post_flag: bool) -> list:
        """
        Fills the array containing information about the entity mention annotations, used in the _text_to_cols method
        :param annot_array: array to be filled
        :param key: reddit id, on which the post title/comment thread is matched with its corresponding annotation
        :param post_flag: flag indicating whether the annotations are collected for the post titles (=True)
        or comment threads (=False)
        """
        next_annot = None
        while True:
            # Check if further annotations belong to the current post title or comment thread as well
            try:
                next_annot = next(self.post_annotations) if post_flag else next(self.comment_annotations)
                if next_annot[0] == key:
                    annot_array.append((int(next_annot[4]), int(next_annot[5]), next_annot[3]))
                else:
                    self.curr_annot = next_annot
                    break
            # Stop when the end of an annotation file is reached
            except StopIteration:
                break
        return annot_array

    def _fill_curr_comment(self, fix_flag: bool):
        """
        Extends the string containing the current comment thread, which is passed to _text_to_cols method, when the
        comments are parsed.
        :param fix_flag: flag indicating whether the method is called when the incorrectly imported rows are parsed (=True)
        or regular rows (=False)
        """
        next_row = None
        while True:
            # Check if further annotations belong to the current sentence as well
            try:
                next_row = next(self.comments) if not fix_flag else next(self.parsed_row)
                if len(next_row) < 2:
                    # 'else "  "' is needed to keep the proper token positions (for accordance with annotations)
                    self.curr_comm += next_row[0] if any(next_row) else "  "
                else:
                    self.curr_row = next_row
                    break
            except StopIteration:  # When the end of the comments.tsv file is reached
                self.curr_row = next_row
                self.stop_iter = not fix_flag
                break


def from_ufsac_to_tsv(
    xml_file: Union[str, Path],
    conll_file: Union[str, Path],
    datasetname: str,
    encoding: str = "utf8",
    cut_multisense: bool = True,
):
    """
    Function that converts the UFSAC format into tab separated column format in a new file.
    Parameters
    ----------
    xml_file : Union[str, Path]
        Path to the xml file.
    conll_file : Union[str, Path]
        Path for the new conll file.
    datasetname: str
        Name of the dataset from UFSAC, needed because of different handling of multi-word-spans in the datasets
    encoding : str, optional
        Encoding used in open function. The default is "utf8".
    cut_multisense : bool, optional
        Boolean that determines whether or not the wn30_key tag should be cut if it contains multiple possible senses.
        If True only the first listed sense will be used. Otherwise the whole list of senses will be detected
        as one new sense. The default is True.

    """

    def make_line(word, begin_or_inside, attributes):
        """
        Function that creates an output line from a word.
        Parameters
        ----------
        word :
            String of the actual word.
        begin_or_inside:
            Either 'B-' or 'I-'
        attributes:
            List of attributes of the word (pos, lemma, wn30_key)
        """
        line = word
        if cut_multisense:
            attributes[-1] = attributes[-1].split(";")[0]  # take only first sense

        for attrib in attributes:
            if attrib != "O":
                line = line + "\t" + begin_or_inside + attrib
            else:
                line = line + "\tO"
        line += "\n"

        return line

    def split_span(word_fields: List[str], datasetname: str):
        """
        Function that splits a word if necessary, i.e. if it is a multiple-word-span.
        Parameters
        ----------
        word_fields :
            list ['surface_form', 'lemma', 'pos', 'wn30_key'] of a word
        datasetname:
            name of corresponding dataset
        """

        span = word_fields[0]

        if datasetname in [
            "trainomatic",
            "masc",
        ]:  # splitting not sensible for these datasets
            return [span]
        elif datasetname == "omsti":
            if (
                word_fields[3] != "O" and not span == "_" and "__" not in span
            ):  # has annotation and does not consist only of '_' (still not 100% clean)
                return span.split("_")
            else:
                return [span]
        else:  # for all other datasets splitting at '_' is always sensible
            return span.split("_")

    txt_out = open(file=conll_file, mode="w", encoding=encoding)
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_file)
    corpus = tree.getroot()

    number_of_docs = len(corpus.findall("document"))

    fields = ["surface_form", "lemma", "pos", "wn30_key"]
    for document in corpus:
        # Docstart
        if number_of_docs > 1:
            txt_out.write("-DOCSTART-\n\n")

        for paragraph in document:

            for sentence in paragraph:

                for word in sentence:

                    dictionary = word.attrib
                    fields_of_word = [word.attrib[field] if (field in dictionary) else "O" for field in fields]

                    chunks = split_span(fields_of_word, datasetname)

                    txt_out.write(make_line(chunks[0], "B-", fields_of_word[1:]))

                    # if there is more than one word in the chunk we write each in a separate line
                    for chunk in chunks[1:]:
                        # print(chunks)
                        txt_out.write(make_line(chunk, "I-", fields_of_word[1:]))

                # empty line after each sentence
                txt_out.write("\n")

    txt_out.close()


def determine_tsv_file(filename: str, data_folder: Path, cut_multisense: bool = True):
    """
    Checks if the converted .tsv file already exists and if not, creates it. Returns name of the file.
    ----------
    string : str
        String that contains the name of the file.
    data_folder : str
        String that contains the name of the folder in which the CoNLL file should reside.
    cut_multisense : bool, optional
        Boolean that determines whether or not the wn30_key tag should be cut if it contains multiple possible senses.
        If True only the first listed sense will be used. Otherwise the whole list of senses will be detected
        as one new sense. The default is True.
    """

    if cut_multisense is True and filename not in [
        "semeval2007task17",
        "trainomatic",
        "wngt",
    ]:  # these three datasets do not have multiple senses

        conll_file_name = filename + "_cut.tsv"

    else:

        conll_file_name = filename + ".tsv"

    path_to_conll_file = data_folder / conll_file_name

    if not path_to_conll_file.exists():
        # convert the file to CoNLL

        from_ufsac_to_tsv(
            xml_file=Path(data_folder / "original_data" / (filename + ".xml")),
            conll_file=Path(data_folder / conll_file_name),
            datasetname=filename,
            cut_multisense=cut_multisense,
        )

    return conll_file_name


class WSD_UFSAC(MultiCorpus):
    def __init__(
        self,
        filenames: Union[str, List[str]] = ["masc", "semcor"],
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        cut_multisense: bool = True,
        columns={0: "text", 3: "sense"},
        tag_to_bioes=None,
        banned_sentences: List[str] = None,
        sample_missing_splits_in_multicorpus: Union[bool, str] = True,
        sample_missing_splits_in_each_corpus: Union[bool, str] = True,
        use_raganato_ALL_as_test_data: bool = False,
        name: str = "multicorpus",
    ):
        """
        Initialize a custom corpus with any Word Sense Disambiguation (WSD) datasets in the UFSAC format from https://github.com/getalp/UFSAC.
        If the constructor is called for the first time the data is automatically downloaded and transformed from xml to a tab separated column format.
        Since only the WordNet 3.0 version for senses is consistently available for all provided datasets we will only consider this version.
        Also we ignore the id annotation used in datasets that were originally created for evaluation tasks
        :param filenames: Here you can pass a single datasetname or a list of ddatasetnames. The available names are:
            'masc', 'omsti', 'raganato_ALL', 'raganato_semeval2007', 'raganato_semeval2013', 'raganato_semeval2015', 'raganato_senseval2', 'raganato_senseval3',
            'semcor', 'semeval2007task17', 'semeval2007task7', 'semeval2013task12', 'semeval2015task13', 'senseval2', 'senseval2_lexical_sample_test',
            'senseval2_lexical_sample_train', 'senseval3task1', 'senseval3task6_test', 'senseval3task6_train', 'trainomatic', 'wngt'.
            So you can pass for example filenames = ['masc', 'omsti', 'wngt']. Default two mid-sized datasets 'masc' and 'semcor' are loaded.
        :param base_path: You can override this to point to a specific folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        :param cut_multisense: Boolean that determines whether or not the wn30_key tag should be cut if it contains
                               multiple possible senses. If True only the first listed sense will be used and the
                               suffix '_cut' will be added to the name of the CoNLL file. Otherwise the whole list of
                               senses will be detected as one new sense. The default is True.
        :param columns: Columns to consider when loading the dataset. You can add 1: "lemma" or 2: "pos" to the default dict {0: "text", 3: "sense"}
            if you want to use additional pos and/or lemma for the words.
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :param banned_sentences: Optionally remove sentences from the corpus. Works only if `in_memory` is true
        :param sample_missing_splits_in_multicorpus: Whether to sample missing splits when loading the multicorpus (this is redundant if
                                                                                                                    sample_missing_splits_in_each_corpus is True)
        :param sample_missing_splits_in_each_corpus: Whether to sample missing splits when loading each single corpus given in filenames.
        :param use_raganato_ALL_as_test_data: If True, the raganato_ALL dataset (Raganato et al. "Word Sense Disambiguation: A unified evaluation framework and empirical compariso")
            will be used as test data. Note that the sample_missing_splits parameters are set to 'only_dev' in this case if set to True.
        :param name: Name of your (costum) corpus
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name
        original_data_folder = data_folder / "original_data"

        # check if data there, if not, download the data
        if not original_data_folder.exists():
            # create folder
            data_folder.mkdir(parents=True)

            # download data
            import gdown

            url = "https://drive.google.com/uc?id=1Oigo3kzRosz2VjyA44vpJZ58tDFyLRMO"

            output = data_folder / (dataset_name + ".tar")

            gdown.download(url, str(output), quiet=False)

            output = data_folder / (dataset_name + ".tar")
            unpack_file(file=output, unpack_to=data_folder, mode="tar", keep=False)

            os.rename(data_folder / "ufsac-public-2.1", original_data_folder)

        # transform data into column format if necessary

        # if no filenames are specified we use all the data
        if not filenames:
            filenames = [name[:-4] for name in os.listdir(original_data_folder) if "raganato" not in name]

        if isinstance(filenames, str):
            filenames = [filenames]

        corpora: List[Corpus] = []

        log.info("Transforming data into column format and creating corpora...")

        if use_raganato_ALL_as_test_data:
            # in this case no test data should be generated by sampling from train data. But if the sample arguments are set to true, the dev set will be sampled
            if sample_missing_splits_in_each_corpus:
                sample_missing_splits_in_each_corpus = "only_dev"
            if sample_missing_splits_in_multicorpus:
                sample_missing_splits_in_multicorpus = "only_dev"

            # also we remove 'raganato_ALL' from filenames in case its in the list
            if "raganato_ALL" in filenames:
                filenames.remove("raganato_ALL")

            # generate the test file
            test_file = determine_tsv_file(
                filename="raganato_ALL",
                data_folder=data_folder,
                cut_multisense=cut_multisense,
            )

            corpus = ColumnCorpus(
                data_folder=data_folder,
                column_format=columns,
                test_file=test_file,  # corpus only has test data
                in_memory=in_memory,
                tag_to_bioes=tag_to_bioes,
                column_delimiter="\t",
                document_separator_token="-DOCSTART-",
                banned_sentences=banned_sentences,
                autofind_splits=False,
                sample_missing_splits=sample_missing_splits_in_each_corpus,
            )
            corpora.append(corpus)

        for filename in filenames:
            # make column file and save to data_folder

            new_filename = determine_tsv_file(
                filename=filename,
                data_folder=data_folder,
                cut_multisense=cut_multisense,
            )

            corpus = ColumnCorpus(
                data_folder=data_folder,
                column_format=columns,
                train_file=new_filename,
                in_memory=in_memory,
                tag_to_bioes=tag_to_bioes,
                column_delimiter="\t",
                document_separator_token="-DOCSTART-",
                banned_sentences=banned_sentences,
                autofind_splits=False,
                sample_missing_splits=sample_missing_splits_in_each_corpus,
            )
            corpora.append(corpus)
        log.info("Done with transforming data into column format and creating corpora...")

        super(WSD_UFSAC, self).__init__(
            corpora,
            sample_missing_splits=sample_missing_splits_in_multicorpus,
            name=name,
        )


class WSD_RAGANATO_ALL(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        columns={0: "text", 3: "sense"},
        tag_to_bioes=None,
        label_name_map: Dict[str, str] = None,
        banned_sentences: List[str] = None,
        sample_missing_splits: bool = True,
        cut_multisense: bool = True,
    ):
        """
        Initialize ragnato_ALL (concatenation of all SensEval and SemEval all-words tasks) provided in UFSAC https://github.com/getalp/UFSAC
        When first initializing the corpus the whole UFSAC data is downloaded.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        dataset_name = "wsd_ufsac"

        data_folder = base_path / dataset_name
        original_data_folder = data_folder / "original_data"

        # We check if the the UFSAC data has already been downloaded. If not, we download it.
        # Note that this downloads more datasets than just SemCor. But the size of the download is only around 190 Mb (around 4.5 Gb unpacked)
        if not original_data_folder.exists():
            # create folder
            data_folder.mkdir(parents=True)

            # download data
            import gdown

            url = "https://drive.google.com/uc?id=1Oigo3kzRosz2VjyA44vpJZ58tDFyLRMO"

            output = data_folder / (dataset_name + ".tar")

            gdown.download(url, str(output), quiet=False)

            output = data_folder / (dataset_name + ".tar")
            unpack_file(file=output, unpack_to=data_folder, mode="tar", keep=False)

            os.rename(data_folder / "ufsac-public-2.1", original_data_folder)

        train_file = determine_tsv_file(
            filename="raganato_ALL",
            data_folder=data_folder,
            cut_multisense=cut_multisense,
        )

        super(WSD_RAGANATO_ALL, self).__init__(
            data_folder=data_folder,
            column_format=columns,
            train_file=train_file,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            column_delimiter="\t",
            autofind_splits=False,
            tag_to_bioes=tag_to_bioes,
            label_name_map=label_name_map,
            banned_sentences=banned_sentences,
            sample_missing_splits=sample_missing_splits,
        )


class WSD_SEMCOR(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        columns={0: "text", 3: "sense"},
        tag_to_bioes=None,
        label_name_map: Dict[str, str] = None,
        banned_sentences: List[str] = None,
        sample_missing_splits: Union[bool, str] = True,
        cut_multisense: bool = True,
        use_raganato_ALL_as_test_data: bool = False,
    ):
        """
        Initialize SemCor provided in UFSAC https://github.com/getalp/UFSAC
        When first initializing the corpus the whole UFSAC data is downloaded.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        dataset_name = "wsd_ufsac"

        data_folder = base_path / dataset_name
        original_data_folder = data_folder / "original_data"

        # We check if the the UFSAC data has already been downloaded. If not, we download it.
        # Note that this downloads more datasets than just SemCor. But the size of the download is only around 190 Mb (around 4.5 Gb unpacked)
        if not original_data_folder.exists():
            # create folder
            data_folder.mkdir(parents=True)

            # download data
            import gdown

            url = "https://drive.google.com/uc?id=1Oigo3kzRosz2VjyA44vpJZ58tDFyLRMO"

            output = data_folder / (dataset_name + ".tar")

            gdown.download(url, str(output), quiet=False)

            output = data_folder / (dataset_name + ".tar")
            unpack_file(file=output, unpack_to=data_folder, mode="tar", keep=False)

            os.rename(data_folder / "ufsac-public-2.1", original_data_folder)

        if use_raganato_ALL_as_test_data:
            # in this case no test data should be generated by sampling from train data. But if sample_missing_splits is true, the dev set will be sampled.
            if sample_missing_splits:
                sample_missing_splits = "only_dev"

            # generate the test file
            test_file = determine_tsv_file(
                filename="raganato_ALL",
                data_folder=data_folder,
                cut_multisense=cut_multisense,
            )
        else:
            test_file = None

        train_file = determine_tsv_file(filename="semcor", data_folder=data_folder, cut_multisense=cut_multisense)

        super(WSD_SEMCOR, self).__init__(
            data_folder=data_folder,
            column_format=columns,
            train_file=train_file,
            test_file=test_file,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            column_delimiter="\t",
            autofind_splits=False,
            tag_to_bioes=tag_to_bioes,
            label_name_map=label_name_map,
            banned_sentences=banned_sentences,
            sample_missing_splits=sample_missing_splits,
        )


class WSD_WORDNET_GLOSS_TAGGED(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        columns={0: "text", 3: "sense"},
        tag_to_bioes=None,
        label_name_map: Dict[str, str] = None,
        banned_sentences: List[str] = None,
        sample_missing_splits: Union[bool, str] = True,
        use_raganato_ALL_as_test_data: bool = False,
    ):
        """
        Initialize Princeton WordNet Gloss Corpus provided in UFSAC https://github.com/getalp/UFSAC
        When first initializing the corpus the whole UFSAC data is downloaded.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        dataset_name = "wsd_ufsac"

        data_folder = base_path / dataset_name
        original_data_folder = data_folder / "original_data"

        # We check if the the UFSAC data has already been downloaded. If not, we download it.
        # Note that this downloads more datasets than just WordNet Gloss Tagged. But the size of the download is only around 190 Mb (around 4.5 Gb unpacked)
        if not original_data_folder.exists():
            # create folder
            data_folder.mkdir(parents=True)

            # download data
            import gdown

            url = "https://drive.google.com/uc?id=1Oigo3kzRosz2VjyA44vpJZ58tDFyLRMO"

            output = data_folder / (dataset_name + ".tar")

            gdown.download(url, str(output), quiet=False)

            output = data_folder / (dataset_name + ".tar")
            unpack_file(file=output, unpack_to=data_folder, mode="tar", keep=False)

            os.rename(data_folder / "ufsac-public-2.1", original_data_folder)

        if use_raganato_ALL_as_test_data:
            # in this case no test data should be generated by sampling from train data. But if sample_missing_splits is true, the dev set will be sampled.
            if sample_missing_splits:
                sample_missing_splits = "only_dev"

            # generate the test file
            test_file = determine_tsv_file(filename="raganato_ALL", data_folder=data_folder, cut_multisense=True)
        else:
            test_file = None

        train_file = determine_tsv_file(
            filename="wngt", data_folder=data_folder, cut_multisense=False
        )  # does not have multisense!

        super(WSD_WORDNET_GLOSS_TAGGED, self).__init__(
            data_folder=data_folder,
            column_format=columns,
            train_file=train_file,
            test_file=test_file,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            column_delimiter="\t",
            autofind_splits=False,
            tag_to_bioes=tag_to_bioes,
            label_name_map=label_name_map,
            banned_sentences=banned_sentences,
            sample_missing_splits=sample_missing_splits,
        )


class WSD_MASC(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        columns={0: "text", 3: "sense"},
        tag_to_bioes=None,
        label_name_map: Dict[str, str] = None,
        banned_sentences: List[str] = None,
        sample_missing_splits: Union[bool, str] = True,
        cut_multisense: bool = True,
        use_raganato_ALL_as_test_data: bool = False,
    ):
        """
        Initialize MASC (Manually Annotated Sub-Corpus) provided in UFSAC https://github.com/getalp/UFSAC
        When first initializing the corpus the whole UFSAC data is downloaded.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        dataset_name = "wsd_ufsac"

        # default dataset folder is the cache root
        data_folder = base_path / dataset_name
        original_data_folder = data_folder / "original_data"

        # We check if the the UFSAC data has already been downloaded. If not, we download it.
        # Note that this downloads more datasets than just MASC. But the size of the download is only around 190 Mb (around 4.5 Gb unpacked)
        if not original_data_folder.exists():
            # create folder
            data_folder.mkdir(parents=True)

            # download data
            import gdown

            url = "https://drive.google.com/uc?id=1Oigo3kzRosz2VjyA44vpJZ58tDFyLRMO"

            output = data_folder / (dataset_name + ".tar")

            gdown.download(url, str(output), quiet=False)

            output = data_folder / (dataset_name + ".tar")
            unpack_file(file=output, unpack_to=data_folder, mode="tar", keep=False)

            os.rename(data_folder / "ufsac-public-2.1", original_data_folder)

        if use_raganato_ALL_as_test_data:
            # in this case no test data should be generated by sampling from train data. But if sample_missing_splits is true, the dev set will be sampled.
            if sample_missing_splits:
                sample_missing_splits = "only_dev"

            # generate the test file
            test_file = determine_tsv_file(
                filename="raganato_ALL",
                data_folder=data_folder,
                cut_multisense=cut_multisense,
            )
        else:
            test_file = None

        train_file = determine_tsv_file(filename="masc", data_folder=data_folder, cut_multisense=cut_multisense)

        super(WSD_MASC, self).__init__(
            data_folder=data_folder,
            column_format=columns,
            train_file=train_file,
            test_file=test_file,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            column_delimiter="\t",
            autofind_splits=False,
            tag_to_bioes=tag_to_bioes,
            label_name_map=label_name_map,
            banned_sentences=banned_sentences,
            sample_missing_splits=sample_missing_splits,
        )


class WSD_OMSTI(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        columns={0: "text", 3: "sense"},
        tag_to_bioes=None,
        label_name_map: Dict[str, str] = None,
        banned_sentences: List[str] = None,
        sample_missing_splits: Union[bool, str] = True,
        cut_multisense: bool = True,
        use_raganato_ALL_as_test_data: bool = False,
    ):
        """
        Initialize OMSTI (One Million Sense-Tagged Instances) provided in UFSAC https://github.com/getalp/UFSAC
        When first initializing the corpus the whole UFSAC data is downloaded.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        dataset_name = "wsd_ufsac"

        # default dataset folder is the cache root

        data_folder = base_path / dataset_name
        original_data_folder = data_folder / "original_data"

        # We check if the the UFSAC data has already been downloaded. If not, we download it.
        # Note that this downloads more datasets than just OMSTI. But the size of the download is only around 190 Mb (around 4.5 Gb unpacked)
        if not original_data_folder.exists():
            # create folder
            data_folder.mkdir(parents=True)

            # download data
            import gdown

            url = "https://drive.google.com/uc?id=1Oigo3kzRosz2VjyA44vpJZ58tDFyLRMO"

            output = data_folder / (dataset_name + ".tar")

            gdown.download(url, str(output), quiet=False)

            output = data_folder / (dataset_name + ".tar")
            unpack_file(file=output, unpack_to=data_folder, mode="tar", keep=False)

            os.rename(data_folder / "ufsac-public-2.1", original_data_folder)

        if use_raganato_ALL_as_test_data:
            # in this case no test data should be generated by sampling from train data. But if sample_missing_splits is true, the dev set will be sampled.
            if sample_missing_splits:
                sample_missing_splits = "only_dev"

            # generate the test file
            test_file = determine_tsv_file(
                filename="raganato_ALL",
                data_folder=data_folder,
                cut_multisense=cut_multisense,
            )
        else:
            test_file = None

        train_file = determine_tsv_file(filename="omsti", data_folder=data_folder, cut_multisense=cut_multisense)

        super(WSD_OMSTI, self).__init__(
            data_folder=data_folder,
            column_format=columns,
            train_file=train_file,
            test_file=test_file,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            column_delimiter="\t",
            autofind_splits=False,
            tag_to_bioes=tag_to_bioes,
            label_name_map=label_name_map,
            banned_sentences=banned_sentences,
            sample_missing_splits=sample_missing_splits,
        )


class WSD_TRAINOMATIC(ColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path] = None,
        in_memory: bool = True,
        columns={0: "text", 3: "sense"},
        tag_to_bioes=None,
        label_name_map: Dict[str, str] = None,
        banned_sentences: List[str] = None,
        sample_missing_splits: Union[bool, str] = True,
        use_raganato_ALL_as_test_data: bool = False,
    ):
        """
        Initialize Train-O-Matic provided in UFSAC https://github.com/getalp/UFSAC
        When first initializing the corpus the whole UFSAC data is downloaded.
        """
        if not base_path:
            base_path = flair.cache_root / "datasets"
        else:
            base_path = Path(base_path)

        dataset_name = "wsd_ufsac"

        # default dataset folder is the cache root

        data_folder = base_path / dataset_name
        original_data_folder = data_folder / "original_data"

        # We check if the the UFSAC data has already been downloaded. If not, we download it.
        # Note that this downloads more datasets than just Train-O-Matic. But the size of the download is only around 190 Mb (around 4.5 Gb unpacked)
        if not original_data_folder.exists():
            # create folder
            data_folder.mkdir(parents=True)

            # download data
            import gdown

            url = "https://drive.google.com/uc?id=1Oigo3kzRosz2VjyA44vpJZ58tDFyLRMO"

            output = data_folder / (dataset_name + ".tar")

            gdown.download(url, str(output), quiet=False)

            output = data_folder / (dataset_name + ".tar")
            unpack_file(file=output, unpack_to=data_folder, mode="tar", keep=False)

            os.rename(data_folder / "ufsac-public-2.1", original_data_folder)

        if use_raganato_ALL_as_test_data:
            # in this case no test data should be generated by sampling from train data. But if sample_missing_splits is true, the dev set will be sampled.
            if sample_missing_splits:
                sample_missing_splits = "only_dev"

            # generate the test file
            test_file = determine_tsv_file(filename="raganato_ALL", data_folder=data_folder, cut_multisense=True)
        else:
            test_file = None

        train_file = determine_tsv_file(
            filename="trainomatic", data_folder=data_folder, cut_multisense=False
        )  # no multisenses

        super(WSD_TRAINOMATIC, self).__init__(
            data_folder=data_folder,
            column_format=columns,
            train_file=train_file,
            test_file=test_file,
            in_memory=in_memory,
            document_separator_token="-DOCSTART-",
            column_delimiter="\t",
            autofind_splits=False,
            tag_to_bioes=tag_to_bioes,
            label_name_map=label_name_map,
            banned_sentences=banned_sentences,
            sample_missing_splits=sample_missing_splits,
        )
