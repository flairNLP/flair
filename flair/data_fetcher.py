from typing import List, Dict, Union
import re
import logging
from enum import Enum
from pathlib import Path

import flair
from flair.data import Sentence, TaggedCorpus, Token, MultiCorpus
from flair.file_utils import cached_path


log = logging.getLogger('flair')


class NLPTask(Enum):
    # conll 2000 column format
    CONLL_2000 = 'conll_2000'

    # conll 03 NER column format
    CONLL_03 = 'conll_03'
    CONLL_03_GERMAN = 'conll_03_german'
    CONLL_03_DUTCH = 'conll_03_dutch'
    CONLL_03_SPANISH = 'conll_03_spanish'

    # WNUT-17
    WNUT_17 = 'wnut_17'

    # -- WikiNER datasets
    WIKINER_ENGLISH = 'wikiner_english'
    WIKINER_GERMAN = 'wikiner_german'
    WIKINER_FRENCH = 'wikiner_french'
    WIKINER_SPANISH = 'wikiner_spanish'
    WIKINER_ITALIAN = 'wikiner_italian'
    WIKINER_DUTCH = 'wikiner_dutch'
    WIKINER_POLISH = 'wikiner_polish'
    WIKINER_PORTUGUESE = 'wikiner_portuguese'
    WIKINER_RUSSIAN = 'wikiner_russian'

    # -- Universal Dependencies
    # Germanic
    UD_ENGLISH = 'ud_english'
    UD_GERMAN = 'ud_german'
    UD_DUTCH = 'ud_dutch'
    # Romance
    UD_FRENCH = 'ud_french'
    UD_ITALIAN = 'ud_italian'
    UD_SPANISH = 'ud_spanish'
    UD_PORTUGUESE = 'ud_portuguese'
    UD_ROMANIAN = 'ud_romanian'
    UD_CATALAN = 'ud_catalan'
    # West-Slavic
    UD_POLISH = 'ud_polish'
    UD_CZECH = 'ud_czech'
    UD_SLOVAK = 'ud_slovak'
    # South-Slavic
    UD_SLOVENIAN = 'ud_slovenian'
    UD_CROATIAN = 'ud_croatian'
    UD_SERBIAN = 'ud_serbian'
    UD_BULGARIAN = 'ud_bulgarian'
    # East-Slavic
    UD_RUSSIAN = 'ud_russian'
    # Scandinavian
    UD_SWEDISH = 'ud_swedish'
    UD_DANISH = 'ud_danish'
    UD_NORWEGIAN = 'ud_norwegian'
    UD_FINNISH = 'ud_finnish'
    # Asian
    UD_ARABIC = 'ud_arabic'
    UD_HEBREW = 'ud_hebrew'
    UD_TURKISH = 'ud_turkish'
    UD_PERSIAN = 'ud_persian'
    UD_HINDI = 'ud_hindi'
    UD_INDONESIAN = 'ud_indonesian'
    UD_JAPANESE = 'ud_japanese'
    UD_CHINESE = 'ud_chinese'
    UD_KOREAN = 'ud_korean'

    # other datasets
    ONTONER = 'ontoner'
    FASHION = 'fashion'
    GERMEVAL = 'germeval'
    SRL = 'srl'
    WSD = 'wsd'
    CONLL_12 = 'conll_12'
    PENN = 'penn'
    ONTONOTES = 'ontonotes'

    # text classification format
    IMDB = 'imdb'
    AG_NEWS = 'ag_news'


class NLPTaskDataFetcher:

    @staticmethod
    def load_corpora(tasks: List[Union[NLPTask, str]], base_path: Path = None) -> MultiCorpus:
        return MultiCorpus([NLPTaskDataFetcher.load_corpus(task, base_path) for task in tasks])

    @staticmethod
    def load_corpus(task: Union[NLPTask, str], base_path: Path = None) -> TaggedCorpus:
        """
        Helper function to fetch a TaggedCorpus for a specific NLPTask. For this to work you need to first download
        and put into the appropriate folder structure the corresponsing NLP task data. The tutorials on
        https://github.com/zalandoresearch/flair give more info on how to do this. Alternatively, you can use this
        code to create your own data fetchers.
        :param task: specification of the NLPTask you wish to get
        :param base_path: path to data folder containing tasks sub folders
        :return: a TaggedCorpus consisting of train, dev and test data
        """

        # first, try to fetch dataset online
        if type(task) is NLPTask:
            NLPTaskDataFetcher.download_dataset(task)

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.file_utils.CACHE_ROOT) / 'datasets'

        # get string value if enum is passed
        task = task.value if type(task) is NLPTask else task

        data_folder = base_path / task.lower()

        # the CoNLL 2000 task on chunking has three columns: text, pos and np (chunk)
        if task == NLPTask.CONLL_2000.value:
            columns = {0: 'text', 1: 'pos', 2: 'np'}

            return NLPTaskDataFetcher.load_column_corpus(data_folder,
                                                         columns,
                                                         tag_to_biloes='np')

        # many NER tasks follow the CoNLL 03 format with four colulms: text, pos, np and ner tag
        if task == NLPTask.CONLL_03.value or task == NLPTask.ONTONER.value or task == NLPTask.FASHION.value:
            columns = {0: 'text', 1: 'pos', 2: 'np', 3: 'ner'}

            return NLPTaskDataFetcher.load_column_corpus(data_folder,
                                                         columns,
                                                         tag_to_biloes='ner',
                                                         )

        # the CoNLL 03 task for German has an additional lemma column
        if task == NLPTask.CONLL_03_GERMAN.value:
            columns = {0: 'text', 1: 'lemma', 2: 'pos', 3: 'np', 4: 'ner'}

            return NLPTaskDataFetcher.load_column_corpus(data_folder,
                                                         columns,
                                                         tag_to_biloes='ner')

        # the CoNLL 03 task for Dutch has no NP column
        if task == NLPTask.CONLL_03_DUTCH.value or task.startswith('wikiner'):
            columns = {0: 'text', 1: 'pos', 2: 'ner'}

            return NLPTaskDataFetcher.load_column_corpus(data_folder,
                                                         columns,
                                                         tag_to_biloes='ner')

        # the CoNLL 03 task for Spanish only has two columns
        if task == NLPTask.CONLL_03_SPANISH.value or task == NLPTask.WNUT_17.value:
            columns = {0: 'text', 1: 'ner'}

            return NLPTaskDataFetcher.load_column_corpus(data_folder,
                                                         columns,
                                                         tag_to_biloes='ner')

        # the GERMEVAL task only has two columns: text and ner
        if task == NLPTask.GERMEVAL.value:
            columns = {1: 'text', 2: 'ner'}

            return NLPTaskDataFetcher.load_column_corpus(data_folder,
                                                         columns,
                                                         tag_to_biloes='ner')

        # WSD tasks may be put into this column format
        if task == NLPTask.WSD.value:
            columns = {0: 'text', 1: 'lemma', 2: 'pos', 3: 'sense'}
            return NLPTaskDataFetcher.load_column_corpus(data_folder,
                                                         columns,
                                                         train_file='semcor.tsv',
                                                         test_file='semeval2015.tsv')

        # the UD corpora follow the CoNLL-U format, for which we have a special reader
        if task.startswith('ud_') or task in [NLPTask.ONTONOTES.value, NLPTask.CONLL_12.value, NLPTask.PENN.value]:
            return NLPTaskDataFetcher.load_ud_corpus(data_folder)

        # for text classifiers, we use our own special format
        if task == NLPTask.IMDB.value or task == NLPTask.AG_NEWS.value:
            return NLPTaskDataFetcher.load_classification_corpus(data_folder)

    @staticmethod
    def load_column_corpus(
            data_folder: Path,
            column_format: Dict[int, str],
            train_file=None,
            test_file=None,
            dev_file=None,
            tag_to_biloes=None) -> TaggedCorpus:
        """
        Helper function to get a TaggedCorpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_biloes: whether to convert to BILOES tagging scheme
        :return: a TaggedCorpus with annotated train, dev and test data
        """
        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if file_name.endswith('.gz'): continue
                if 'train' in file_name and not '54019' in file_name:
                    train_file = file
                if 'test' in file_name:
                    test_file = file
                if 'dev' in file_name:
                    dev_file = file
                if 'testa' in file_name:
                    dev_file = file
                if 'testb' in file_name:
                    test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        # get train and test data
        sentences_train: List[Sentence] = NLPTaskDataFetcher.read_column_data(train_file, column_format)

        # read in test file if exists, otherwise sample 10% of train data as test dataset
        if test_file is not None:
            sentences_test: List[Sentence] = NLPTaskDataFetcher.read_column_data(test_file, column_format)
        else:
            sentences_test: List[Sentence] = [sentences_train[i] for i in
                                             NLPTaskDataFetcher.__sample(len(sentences_train), 0.1)]
            sentences_train = [x for x in sentences_train if x not in sentences_test]

        # read in dev file if exists, otherwise sample 10% of train data as dev dataset
        if dev_file is not None:
            sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_column_data(dev_file, column_format)
        else:
            sentences_dev: List[Sentence] = [sentences_train[i] for i in
                                             NLPTaskDataFetcher.__sample(len(sentences_train), 0.1)]
            sentences_train = [x for x in sentences_train if x not in sentences_dev]

        if tag_to_biloes is not None:
            # convert tag scheme to iobes
            for sentence in sentences_train + sentences_test + sentences_dev:
                sentence: Sentence = sentence
                sentence.convert_tag_scheme(tag_type=tag_to_biloes, target_scheme='iobes')

        return TaggedCorpus(sentences_train, sentences_dev, sentences_test, name=data_folder.name)

    @staticmethod
    def load_ud_corpus(
            data_folder: Path,
            train_file=None,
            test_file=None,
            dev_file=None) -> TaggedCorpus:
        """
        Helper function to get a TaggedCorpus from CoNLL-U column-formatted task data such as the UD corpora

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :return: a TaggedCorpus with annotated train, dev and test data
        """
        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if 'train' in file_name:
                    train_file = file
                if 'test' in file_name:
                    test_file = file
                if 'dev' in file_name:
                    dev_file = file
                if 'testa' in file_name:
                    dev_file = file
                if 'testb' in file_name:
                    test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(train_file)
        sentences_test: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(test_file)
        sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(dev_file)

        return TaggedCorpus(sentences_train, sentences_dev, sentences_test, name=data_folder.name)

    @staticmethod
    def load_classification_corpus(
            data_folder: Path,
            train_file=None,
            test_file=None,
            dev_file=None) -> TaggedCorpus:
        """
        Helper function to get a TaggedCorpus from text classification-formatted task data

        :param data_folder: base folder with the task data
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :return: a TaggedCorpus with annotated train, dev and test data
        """
        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if 'train' in file_name:
                    train_file = file
                if 'test' in file_name:
                    test_file = file
                if 'dev' in file_name:
                    dev_file = file
                if 'testa' in file_name:
                    dev_file = file
                if 'testb' in file_name:
                    test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        sentences_train: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file(train_file)
        sentences_test: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file(test_file)
        sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file(dev_file)

        return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

    @staticmethod
    def read_text_classification_file(path_to_file: Path, max_tokens_per_doc=-1) -> List[Sentence]:
        """
        Reads a data file for text classification. The file should contain one document/text per line.
        The line should have the following format:
        __label__<class_name> <text>
        If you have a multi class task, you can have as many labels as you want at the beginning of the line, e.g.,
        __label__<class_name_1> __label__<class_name_2> <text>
        :param path_to_file: the path to the data file
        :param max_tokens_per_doc: Take only documents that contain number of tokens less or equal to this value. If
        set to -1 all documents are taken.
        :return: list of sentences
        """
        label_prefix = '__label__'
        sentences = []

        with open(path_to_file) as f:
            for line in f:
                words = line.split()

                labels = []
                l_len = 0

                for i in range(len(words)):
                    if words[i].startswith(label_prefix):
                        l_len += len(words[i]) + 1
                        label = words[i].replace(label_prefix, "")
                        labels.append(label)
                    else:
                        break

                text = line[l_len:].strip()

                if text and labels:
                    sentence = Sentence(text, labels=labels, use_tokenizer=True)
                    if len(sentence) > max_tokens_per_doc:
                        sentence.tokens = sentence.tokens[:max_tokens_per_doc]
                    if len(sentence.tokens) > 0:
                        sentences.append(sentence)

        return sentences

    @staticmethod
    def read_column_data(path_to_column_file: Path,
                         column_name_map: Dict[int, str],
                         infer_whitespace_after: bool = True):
        """
        Reads a file in column format and produces a list of Sentence with tokenlevel annotation as specified in the
        column_name_map. For instance, by passing "{0: 'text', 1: 'pos', 2: 'np', 3: 'ner'}" as column_name_map you
        specify that the first column is the text (lexical value) of the token, the second the PoS tag, the third
        the chunk and the forth the NER tag.
        :param path_to_column_file: the path to the column file
        :param column_name_map: a map of column number to token annotation name
        :param infer_whitespace_after: if True, tries to infer whitespace_after field for Token
        :return: list of sentences
        """
        sentences: List[Sentence] = []

        try:
            lines: List[str] = open(path_to_column_file, encoding='utf-8').read().strip().split('\n')
        except:
            log.info('UTF-8 can\'t read: {} ... using "latin-1" instead.'.format(path_to_column_file))
            lines: List[str] = open(path_to_column_file, encoding='latin1').read().strip().split('\n')

        # most data sets have the token text in the first column, if not, pass 'text' as column
        text_column: int = 0
        for column in column_name_map:
            if column_name_map[column] == 'text':
                text_column = column

        sentence: Sentence = Sentence()
        for line in lines:

            if line.startswith('#'):
                continue

            if line.strip().replace('﻿', '') == '':
                if len(sentence) > 0:
                    sentence.infer_space_after()
                    sentences.append(sentence)
                sentence: Sentence = Sentence()

            else:
                fields: List[str] = re.split("\s+", line)
                token = Token(fields[text_column])
                for column in column_name_map:
                    if len(fields) > column:
                        if column != text_column:
                            token.add_tag(column_name_map[column], fields[column])

                sentence.add_token(token)

        if len(sentence.tokens) > 0:
            sentence.infer_space_after()
            sentences.append(sentence)

        return sentences

    @staticmethod
    def read_conll_ud(path_to_conll_file: Path) -> List[Sentence]:
        """
       Reads a file in CoNLL-U format and produces a list of Sentence with full morphosyntactic annotation
       :param path_to_conll_file: the path to the conll-u file
       :return: list of sentences
       """
        sentences: List[Sentence] = []

        lines: List[str] = open(path_to_conll_file, encoding='utf-8'). \
            read().strip().split('\n')

        sentence: Sentence = Sentence()
        for line in lines:

            fields: List[str] = re.split("\t+", line)
            if line == '':
                if len(sentence) > 0:
                    sentences.append(sentence)
                sentence: Sentence = Sentence()

            elif line.startswith('#'):
                continue
            elif '.' in fields[0]:
                continue
            elif '-' in fields[0]:
                continue
            else:
                token = Token(fields[1], head_id=int(fields[6]))
                token.add_tag('lemma', str(fields[2]))
                token.add_tag('upos', str(fields[3]))
                token.add_tag('pos', str(fields[4]))
                token.add_tag('dependency', str(fields[7]))

                for morph in str(fields[5]).split('|'):
                    if not "=" in morph: continue;
                    token.add_tag(morph.split('=')[0].lower(), morph.split('=')[1])

                if len(fields) > 10 and str(fields[10]) == 'Y':
                    token.add_tag('frame', str(fields[11]))

                sentence.add_token(token)

        if len(sentence.tokens) > 0: sentences.append(sentence)

        return sentences

    @staticmethod
    def __sample(total_number_of_sentences: int, percentage: float = 0.1) -> List[int]:
        import random
        sample_size: int = round(total_number_of_sentences * percentage)
        sample = random.sample(range(1, total_number_of_sentences), sample_size)
        return sample

    @staticmethod
    def download_dataset(task: NLPTask):

        # conll 2000 chunking task
        if task == NLPTask.CONLL_2000:
            conll_2000_path = 'https://www.clips.uantwerpen.be/conll2000/chunking/'
            data_file = Path(flair.file_utils.CACHE_ROOT) / 'datasets' / task.value / 'train.txt'
            if not data_file.is_file():

                cached_path(f'{conll_2000_path}train.txt.gz', Path('datasets') / task.value)
                cached_path(f'{conll_2000_path}test.txt.gz', Path('datasets') / task.value)
                import gzip, shutil
                with gzip.open(Path(flair.file_utils.CACHE_ROOT) / 'datasets' / task.value / 'train.txt.gz', 'rb') as f_in:
                    with open(Path(flair.file_utils.CACHE_ROOT) / 'datasets' / task.value / 'train.txt', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                with gzip.open(Path(flair.file_utils.CACHE_ROOT) / 'datasets' / task.value / 'test.txt.gz', 'rb') as f_in:
                    with open(Path(flair.file_utils.CACHE_ROOT) / 'datasets' / task.value / 'test.txt', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

        if task == NLPTask.WNUT_17:
            wnut_path = 'https://noisy-text.github.io/2017/files/'
            cached_path(f'{wnut_path}wnut17train.conll', Path('datasets') / task.value)
            cached_path(f'{wnut_path}emerging.dev.conll', Path('datasets') / task.value)
            cached_path(f'{wnut_path}emerging.test.annotated', Path('datasets') / task.value)

        # Wikiner NER task
        wikiner_path = 'https://raw.githubusercontent.com/dice-group/FOX/master/input/Wikiner/'
        if task.value.startswith('wikiner'):
            lc = ''
            if task == NLPTask.WIKINER_ENGLISH: lc = 'en'
            if task == NLPTask.WIKINER_GERMAN: lc = 'de'
            if task == NLPTask.WIKINER_DUTCH: lc = 'nl'
            if task == NLPTask.WIKINER_FRENCH: lc = 'fr'
            if task == NLPTask.WIKINER_ITALIAN: lc = 'it'
            if task == NLPTask.WIKINER_SPANISH: lc = 'es'
            if task == NLPTask.WIKINER_PORTUGUESE: lc = 'pt'
            if task == NLPTask.WIKINER_POLISH: lc = 'pl'
            if task == NLPTask.WIKINER_RUSSIAN: lc = 'ru'

            data_file = Path(flair.file_utils.CACHE_ROOT) / 'datasets' / task.value / f'aij-wikiner-{lc}-wp3.train'
            if not data_file.is_file():

                cached_path(f'{wikiner_path}aij-wikiner-{lc}-wp3.bz2', Path('datasets') / task.value)
                import bz2, shutil

                # unpack and write out in CoNLL column-like format
                bz_file = bz2.BZ2File(
                    Path(flair.file_utils.CACHE_ROOT) / 'datasets' / task.value / f'aij-wikiner-{lc}-wp3.bz2', 'rb')
                with bz_file as f, open(
                        Path(flair.file_utils.CACHE_ROOT) / 'datasets' / task.value / f'aij-wikiner-{lc}-wp3.train',
                        'w') as out:
                    for line in f:
                        line = line.decode('utf-8')
                        words = line.split(' ')
                        for word in words:
                            out.write('\t'.join(word.split('|')) + '\n')

        # CoNLL 02/03 NER
        conll_02_path = 'https://www.clips.uantwerpen.be/conll2002/ner/data/'
        if task == NLPTask.CONLL_03_DUTCH:
            cached_path(f'{conll_02_path}ned.testa', Path('datasets') / task.value)
            cached_path(f'{conll_02_path}ned.testb', Path('datasets') / task.value)
            cached_path(f'{conll_02_path}ned.train', Path('datasets') / task.value)
        if task == NLPTask.CONLL_03_SPANISH:
            cached_path(f'{conll_02_path}esp.testa', Path('datasets') / task.value)
            cached_path(f'{conll_02_path}esp.testb', Path('datasets') / task.value)
            cached_path(f'{conll_02_path}esp.train', Path('datasets') / task.value)

        # universal dependencies
        ud_path = 'https://raw.githubusercontent.com/UniversalDependencies/'
        # --- UD Germanic
        if task == NLPTask.UD_ENGLISH:
            cached_path(f'{ud_path}UD_English-EWT/master/en_ewt-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_English-EWT/master/en_ewt-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_English-EWT/master/en_ewt-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_GERMAN:
            cached_path(f'{ud_path}UD_German-GSD/master/de_gsd-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_German-GSD/master/de_gsd-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_German-GSD/master/de_gsd-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_DUTCH:
            cached_path(f'{ud_path}UD_Dutch-Alpino/master/nl_alpino-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Dutch-Alpino/master/nl_alpino-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Dutch-Alpino/master/nl_alpino-ud-train.conllu', Path('datasets') / task.value)

        # --- UD Romance
        if task == NLPTask.UD_FRENCH:
            cached_path(f'{ud_path}UD_French-GSD/master/fr_gsd-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_French-GSD/master/fr_gsd-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_French-GSD/master/fr_gsd-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_ITALIAN:
            cached_path(f'{ud_path}UD_Italian-ISDT/master/it_isdt-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Italian-ISDT/master/it_isdt-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Italian-ISDT/master/it_isdt-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_SPANISH:
            cached_path(f'{ud_path}UD_Spanish-GSD/master/es_gsd-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Spanish-GSD/master/es_gsd-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Spanish-GSD/master/es_gsd-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_PORTUGUESE:
            cached_path(f'{ud_path}UD_Portuguese-Bosque/blob/master/pt_bosque-ud-dev.conllu',
                        Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Portuguese-Bosque/blob/master/pt_bosque-ud-test.conllu',
                        Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Portuguese-Bosque/blob/master/pt_bosque-ud-train.conllu',
                        Path('datasets') / task.value)

        if task == NLPTask.UD_ROMANIAN:
            cached_path(f'{ud_path}UD_Romanian-RRT/master/ro_rrt-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Romanian-RRT/master/ro_rrt-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Romanian-RRT/master/ro_rrt-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_CATALAN:
            cached_path(f'{ud_path}UD_Catalan-AnCora/master/ca_ancora-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Catalan-AnCora/master/ca_ancora-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Catalan-AnCora/master/ca_ancora-ud-train.conllu', Path('datasets') / task.value)

        # --- UD West-Slavic
        if task == NLPTask.UD_POLISH:
            cached_path(f'{ud_path}UD_Polish-LFG/master/pl_lfg-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Polish-LFG/master/pl_lfg-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Polish-LFG/master/pl_lfg-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_CZECH:
            cached_path(f'{ud_path}UD_Czech-PDT/master/cs_pdt-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Czech-PDT/master/cs_pdt-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Czech-PDT/master/cs_pdt-ud-train-l.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_SLOVAK:
            cached_path(f'{ud_path}UD_Slovak-SNK/master/sk_snk-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Slovak-SNK/master/sk_snk-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Slovak-SNK/master/sk_snk-ud-train.conllu', Path('datasets') / task.value)

        # --- UD Scandinavian
        if task == NLPTask.UD_SWEDISH:
            cached_path(f'{ud_path}UD_Swedish-Talbanken/master/sv_talbanken-ud-dev.conllu',
                        Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Swedish-Talbanken/master/sv_talbanken-ud-test.conllu',
                        Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Swedish-Talbanken/master/sv_talbanken-ud-train.conllu',
                        Path('datasets') / task.value)

        if task == NLPTask.UD_DANISH:
            cached_path(f'{ud_path}UD_Danish-DDT/master/da_ddt-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Danish-DDT/master/da_ddt-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Danish-DDT/master/da_ddt-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_NORWEGIAN:
            cached_path(f'{ud_path}UD_Norwegian-Bokmaal/master/no_bokmaal-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Norwegian-Bokmaal/master/no_bokmaal-ud-test.conllu',
                        Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Norwegian-Bokmaal/master/no_bokmaal-ud-train.conllu',
                        Path('datasets') / task.value)

        if task == NLPTask.UD_FINNISH:
            cached_path(f'{ud_path}UD_Finnish-TDT/master/fi_tdt-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Finnish-TDT/master/fi_tdt-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Finnish-TDT/master/fi_tdt-ud-train.conllu', Path('datasets') / task.value)

        # --- UD South-Slavic
        if task == NLPTask.UD_SLOVENIAN:
            cached_path(f'{ud_path}UD_Slovenian-SSJ/master/sl_ssj-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Slovenian-SSJ/master/sl_ssj-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Slovenian-SSJ/master/sl_ssj-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_CROATIAN:
            cached_path(f'{ud_path}UD_Croatian-SET/master/hr_set-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Croatian-SET/master/hr_set-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Croatian-SET/master/hr_set-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_SERBIAN:
            cached_path(f'{ud_path}UD_Serbian-SET/master/sr_set-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Serbian-SET/master/sr_set-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Serbian-SET/master/sr_set-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_BULGARIAN:
            cached_path(f'{ud_path}UD_Bulgarian-BTB/master/bg_btb-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Bulgarian-BTB/master/bg_btb-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Bulgarian-BTB/master/bg_btb-ud-train.conllu', Path('datasets') / task.value)

        # --- UD Asian
        if task == NLPTask.UD_ARABIC:
            cached_path(f'{ud_path}UD_Arabic-PADT/master/ar_padt-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Arabic-PADT/master/ar_padt-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Arabic-PADT/master/ar_padt-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_HEBREW:
            cached_path(f'{ud_path}UD_Hebrew-HTB/master/he_htb-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Hebrew-HTB/master/he_htb-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Hebrew-HTB/master/he_htb-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_TURKISH:
            cached_path(f'{ud_path}UD_Turkish-IMST/master/tr_imst-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Turkish-IMST/master/tr_imst-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Turkish-IMST/master/tr_imst-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_PERSIAN:
            cached_path(f'{ud_path}UD_Persian-Seraji/master/fa_seraji-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Persian-Seraji/master/fa_seraji-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Persian-Seraji/master/fa_seraji-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_RUSSIAN:
            cached_path(f'{ud_path}UD_Russian-SynTagRus/master/ru_syntagrus-ud-dev.conllu',
                        Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Russian-SynTagRus/master/ru_syntagrus-ud-test.conllu',
                        Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Russian-SynTagRus/master/ru_syntagrus-ud-train.conllu',
                        Path('datasets') / task.value)

        if task == NLPTask.UD_HINDI:
            cached_path(f'{ud_path}UD_Hindi-HDTB/master/hi_hdtb-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Hindi-HDTB/master/hi_hdtb-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Hindi-HDTB/master/hi_hdtb-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_INDONESIAN:
            cached_path(f'{ud_path}UD_Indonesian-GSD/master/id_gsd-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Indonesian-GSD/master/id_gsd-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Indonesian-GSD/master/id_gsd-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_JAPANESE:
            cached_path(f'{ud_path}UD_Japanese-GSD/master/ja_gsd-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Japanese-GSD/master/ja_gsd-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Japanese-GSD/master/ja_gsd-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_CHINESE:
            cached_path(f'{ud_path}UD_Chinese-GSD/master/zh_gsd-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Chinese-GSD/master/zh_gsd-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Chinese-GSD/master/zh_gsd-ud-train.conllu', Path('datasets') / task.value)

        if task == NLPTask.UD_KOREAN:
            cached_path(f'{ud_path}UD_Korean-Kaist/master/ko_kaist-ud-dev.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Korean-Kaist/master/ko_kaist-ud-test.conllu', Path('datasets') / task.value)
            cached_path(f'{ud_path}UD_Korean-Kaist/master/ko_kaist-ud-train.conllu', Path('datasets') / task.value)
