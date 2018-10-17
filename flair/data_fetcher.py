from typing import List, Dict
import re
import os
import logging
from enum import Enum

from flair.data import Sentence, TaggedCorpus, Token

log = logging.getLogger(__name__)


class NLPTask(Enum):
    # conll column format
    CONLL_03 = 'conll_03'
    CONLL_2000 = 'conll_2000'
    CONLL_03_GERMAN = 'conll_03-ger'
    ONTONER = 'onto-ner'
    FASHION = 'fashion'
    GERMEVAL = 'germeval'
    SRL = 'srl'
    WSD = 'wsd'

    # conll-u format
    UD_ENGLISH = 'ud_english'
    UD_GERMAN = 'ud_german'
    CONLL_12 = 'conll_12'
    PENN = 'penn'
    ONTONOTES = 'ontonotes'

    # text classification format
    IMDB = 'imdb'
    AG_NEWS = 'ag_news'


class NLPTaskDataFetcher:

    @staticmethod
    def fetch_data(task: NLPTask) -> TaggedCorpus:
        """
        Helper function to fetch a TaggedCorpus for a specific NLPTask. For this to work you need to first download
        and put into the appropriate folder structure the corresponsing NLP task data. The tutorials on
        https://github.com/zalandoresearch/flair give more info on how to do this. Alternatively, you can use this
        code to create your own data fetchers.
        :param task: specification of the NLPTask you wish to get
        :return: a TaggedCorpus consisting of train, dev and test data
        """

        data_folder = os.path.join('resources', 'tasks', str(task.value).lower())
        log.info("Reading data from {}".format(data_folder))

        # the CoNLL 2000 task on chunking has three columns: text, pos and np (chunk)
        if task == NLPTask.CONLL_2000:
            columns = {0: 'text', 1: 'pos', 2: 'np'}

            return NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                          columns,
                                                          train_file='train.txt',
                                                          test_file='test.txt',
                                                          tag_to_biloes='np')

        # many NER tasks follow the CoNLL 03 format with four colulms: text, pos, np and ner tag
        if task == NLPTask.CONLL_03 or task == NLPTask.ONTONER or task == NLPTask.FASHION:
            columns = {0: 'text', 1: 'pos', 2: 'np', 3: 'ner'}

            return NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                          columns,
                                                          train_file='eng.train',
                                                          test_file='eng.testb',
                                                          dev_file='eng.testa',
                                                          tag_to_biloes='ner',
                                                          )

        # the CoNLL 03 task for German has an additional lemma column
        if task == NLPTask.CONLL_03_GERMAN:
            columns = {0: 'text', 1: 'lemma', 2: 'pos', 3: 'np', 4: 'ner'}

            return NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                          columns,
                                                          train_file='deu.train',
                                                          test_file='deu.testb',
                                                          dev_file='deu.testa',
                                                          tag_to_biloes='ner')

        # the GERMEVAL task only has two columns: text and ner
        if task == NLPTask.GERMEVAL:
            columns = {1: 'text', 2: 'ner'}

            return NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                          columns,
                                                          train_file='NER-de-train.tsv',
                                                          test_file='NER-de-test.tsv',
                                                          dev_file='NER-de-dev.tsv',
                                                          tag_to_biloes='ner')

        # WSD tasks may be put into this column format
        if task == NLPTask.WSD:
            columns = {0: 'text', 1: 'lemma', 2: 'pos', 3: 'sense'}

            return NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                          columns,
                                                          train_file='semcor.tsv',
                                                          test_file='semeval2015.tsv')

        # the UD corpora follow the CoNLL-U format, for which we have a special reader
        if task == NLPTask.UD_ENGLISH:
            # get train, test and dev data
            sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'en_ewt-ud-train.conllu'))
            sentences_test: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'en_ewt-ud-test.conllu'))
            sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'en_ewt-ud-dev.conllu'))

            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        if task == NLPTask.UD_GERMAN:
            # get train, test and dev data
            sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'de_gsd-ud-train.conllu'))
            sentences_test: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'de_gsd-ud-test.conllu'))
            sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'de_gsd-ud-dev.conllu'))

            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        if task == NLPTask.ONTONOTES:
            # get train, test and dev data
            sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'train.conllu'))
            sentences_test: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'test.conllu'))
            sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'dev.conllu'))

            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        if task == NLPTask.CONLL_12:
            # get train, test and dev data
            sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'train.propbank.conllu'))
            sentences_test: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'test.propbank.conllu'))
            sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'dev.propbank.conllu'))
            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        if task == NLPTask.PENN:
            sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'train.conll'))
            sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'valid.conll'))
            sentences_test: List[Sentence] = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'test.conll'))

            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        # for text classifiers, we use our own special format
        if task == NLPTask.IMDB:
            sentences_train: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'train.txt'))
            sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'dev.txt'))
            sentences_test: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'test.txt'))
            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        # for text classifiers, we use our own special format
        if task == NLPTask.AG_NEWS:
            sentences_train: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'train.txt'))
            sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'dev.txt'))
            sentences_test: List[Sentence] = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'test.txt'))
            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

    @staticmethod
    def fetch_column_corpus(
            data_folder: str,
            column_format: Dict[int, str],
            train_file: str,
            test_file: str,
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

        # get train and test data
        sentences_train: List[Sentence] = NLPTaskDataFetcher.read_column_data(
            os.path.join(data_folder, train_file), column_format)
        sentences_test: List[Sentence] = NLPTaskDataFetcher.read_column_data(
            os.path.join(data_folder, test_file), column_format)

        if dev_file is not None:
            sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_column_data(
                os.path.join(data_folder, dev_file), column_format)
        else:
            # sample dev data from train
            sentences_dev: List[Sentence] = [sentences_train[i] for i in NLPTaskDataFetcher.__sample()]
            sentences_train = [x for x in sentences_train if x not in sentences_dev]

        if tag_to_biloes is not None:
            # convert tag scheme to iobes
            for sentence in sentences_train + sentences_test + sentences_dev:
                sentence: Sentence = sentence
                sentence.convert_tag_scheme(tag_type=tag_to_biloes, target_scheme='iobes')

        return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

    @staticmethod
    def read_column_data(path_to_column_file: str,
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

        lines: List[str] = open(path_to_column_file).read().strip().split('\n')

        # most data sets have the token text in the first column, if not, pass 'text' as column
        text_column: int = 0
        for column in column_name_map:
            if column_name_map[column] == 'text':
                text_column = column

        sentence: Sentence = Sentence()
        for line in lines:

            if line.startswith('#'):
                continue

            if line == '':
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
    def read_conll_ud(path_to_conll_file: str) -> List[Sentence]:
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

            fields: List[str] = re.split("\s+", line)
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
    def read_text_classification_file(path_to_file, max_tokens_per_doc=-1):
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
                    if(len(sentence) > max_tokens_per_doc):
                        sentence.tokens = sentence.tokens[:max_tokens_per_doc]
                    sentences.append(sentence)

        return sentences

    @staticmethod
    def __sample():
        sample = [7199, 2012, 7426, 1374, 2590, 4401, 7659, 2441, 4209, 6997, 6907, 4789, 3292, 4874, 7836, 2065, 1804,
                  2409,
                  6353, 86, 1412, 5431, 3275, 7696, 3325, 7678, 6888, 5326, 5782, 3739, 4972, 6350, 7167, 6558, 918,
                  6444,
                  5368, 731, 244, 2029, 6200, 5088, 4688, 2580, 2153, 5477, 714, 1570, 6651, 5724, 4090, 167, 1689,
                  6166,
                  7304, 3705, 256, 5689, 6282, 707, 5390, 1367, 4167, 16, 6554, 5093, 3944, 5008, 3510, 1741, 1, 4464,
                  173,
                  5362, 6827, 35, 1662, 3136, 1516, 3826, 1575, 6771, 5965, 1449, 7806, 632, 5870, 3566, 1434, 2361,
                  6348,
                  5140, 7765, 4800, 6541, 7910, 2021, 1041, 3171, 2137, 495, 2249, 7334, 4806, 844, 3848, 7396, 3861,
                  1337,
                  430, 1325, 36, 2289, 720, 4182, 3955, 3451, 192, 3715, 3144, 1140, 2874, 6728, 4877, 1876, 2551, 2910,
                  260,
                  7767, 7206, 5577, 6707, 3392, 1830, 842, 5264, 4042, 3572, 331, 6995, 2307, 5664, 2878, 1115, 1880,
                  1548,
                  3740, 860, 1799, 2099, 7359, 4648, 2264, 1018, 5417, 3052, 2480, 2256, 6672, 6647, 1272, 1986, 7063,
                  4071,
                  3199, 3652, 1797, 1693, 2008, 4138, 7428, 3083, 1494, 4911, 728, 1556, 7651, 2535, 2160, 4014, 1438,
                  6148,
                  551, 476, 4198, 3835, 1489, 6404, 7346, 1178, 607, 7693, 4146, 6655, 4355, 1571, 522, 5835, 622, 1267,
                  6778, 5236, 5211, 5039, 3836, 1751, 1019, 6952, 7610, 7677, 4224, 1485, 4101, 5793, 6708, 5741, 4630,
                  5857,
                  6959, 847, 4375, 3458, 4936, 6887, 5, 3150, 5551, 4840, 2618, 7456, 7600, 5995, 5270, 5496, 4316,
                  1479,
                  517, 2940, 2337, 7461, 3296, 4133, 491, 6408, 7609, 4290, 5028, 7471, 6337, 488, 5033, 5967, 1209,
                  5511,
                  5449, 3837, 4760, 4490, 6550, 2676, 371, 3962, 4507, 5268, 4285, 5257, 859, 14, 4487, 5669, 6594,
                  6544,
                  7427, 5624, 4882, 7425, 2378, 1498, 931, 7253, 2638, 2897, 5670, 6463, 5300, 6802, 4229, 7076, 6848,
                  6414,
                  1465, 7243, 989, 7204, 1926, 1255, 1794, 2115, 3975, 6987, 3166, 105, 3856, 3272, 3977, 4097, 2612,
                  2869,
                  6022, 153, 3357, 2439, 6491, 766, 3840, 2683, 5074, 159, 5407, 3029, 4815, 1782, 4970, 6250, 5377,
                  6473,
                  5151, 4687, 798, 5214, 3364, 6412, 7125, 3495, 2385, 4476, 863, 5493, 5830, 938, 2979, 7808, 4830,
                  4180,
                  1565, 4818, 702, 1442, 4673, 6920, 2089, 1930, 2036, 1436, 6632, 1006, 5256, 5666, 6401, 3415, 4693,
                  5890,
                  7124, 3853, 884, 4650, 4550, 7406, 3394, 6715, 6754, 3932, 599, 1816, 3273, 5016, 2918, 526, 6883,
                  3089,
                  64, 1305, 7442, 6837, 783, 4536, 100, 4951, 2933, 3750, 3232, 7150, 1934, 3576, 2900, 7883, 964, 4025,
                  28,
                  1732, 382, 166, 6053, 6320, 2058, 652, 3182, 6836, 4547, 419, 1600, 6891, 6235, 7208, 7190, 7144,
                  3133,
                  4775, 4892, 895, 4428, 7929, 7297, 7773, 5325, 2799, 5645, 1192, 1672, 2540, 6812, 5441, 2681, 342,
                  333,
                  2161, 593, 5463, 1568, 5252, 4194, 2280, 2423, 2118, 7455, 4553, 5960, 3163, 7147, 4305, 5599, 2775,
                  5334,
                  4727, 6926, 2189, 7778, 7245, 2066, 1259, 2074, 7866, 7403, 4642, 5490, 3563, 6923, 3934, 5728, 5425,
                  2369,
                  375, 3578, 2732, 2675, 6167, 6726, 4211, 2241, 4585, 4272, 882, 1821, 3904, 6864, 5723, 4708, 3226,
                  7151,
                  3911, 4274, 4945, 3719, 7467, 7712, 5068, 7181, 745, 2846, 2695, 3707, 1076, 1077, 2698, 5699, 1040,
                  6338,
                  631, 1609, 896, 3607, 6801, 3593, 1698, 91, 639, 2826, 2937, 493, 4218, 5958, 2765, 4926, 4546, 7400,
                  1909,
                  5693, 1871, 1687, 6589, 4334, 2748, 7129, 3332, 42, 345, 709, 4685, 6624, 377, 3204, 2603, 7183, 6123,
                  4249, 1531, 7, 703, 6978, 2856, 7871, 7290, 369, 582, 4704, 4979, 66, 1139, 87, 5166, 967, 2727, 5920,
                  6806, 5997, 1301, 5826, 1805, 4347, 4870, 4213, 4254, 504, 3865, 189, 6393, 7281, 2907, 656, 6617,
                  1807,
                  6258, 3605, 1009, 3694, 3004, 2870, 7710, 2608, 400, 7635, 4392, 3055, 942, 2952, 3441, 902, 5892,
                  574,
                  5418, 6212, 1602, 5619, 7094, 1168, 3877, 3888, 1618, 6564, 455, 4581, 3258, 2606, 4643, 2454, 2763,
                  5332,
                  6158, 940, 2343, 7902, 3438, 6117, 2198, 3842, 4773, 1492, 2424, 7662, 6559, 1196, 3203, 5286, 6764,
                  3829,
                  4746, 1117, 2120, 1378, 5614, 4871, 4024, 5489, 3312, 1094, 1838, 3964, 3151, 4545, 5795, 1739, 4920,
                  5690,
                  2570, 3530, 2751, 1426, 2631, 88, 7728, 3741, 5654, 3157, 5557, 6668, 7309, 7313, 807, 4376, 4512,
                  6786,
                  7898, 2429, 3890, 2418, 2243, 2330, 4561, 6119, 2864, 5570, 2485, 5499, 4983, 6257, 3692, 1563, 1939,
                  126,
                  3299, 2811, 7933, 465, 5976, 3712, 4478, 7671, 3143, 1947, 6133, 1928, 5725, 5747, 1107, 163, 3610,
                  3723,
                  1496, 7477, 53, 6548, 5548, 4357, 4963, 5896, 5361, 7295, 7632, 3559, 6740, 6312, 6890, 3303, 625,
                  7681,
                  7174, 6928, 1088, 2133, 4276, 5299, 4488, 5354, 3044, 3321, 409, 6218, 2255, 829, 2129, 673, 1588,
                  6824,
                  1297, 6996, 4324, 7423, 5209, 7617, 3041, 78, 5518, 5392, 4967, 3704, 497, 858, 1833, 5108, 6095,
                  6039,
                  6705, 5561, 5888, 3883, 1048, 1119, 1292, 5639, 4358, 2487, 1235, 125, 4453, 3035, 3304, 6938, 2670,
                  4322,
                  648, 1785, 6114, 6056, 1515, 4628, 5036, 37, 1226, 6081, 4473, 953, 5009, 217, 5952, 755, 2604, 3060,
                  3322,
                  6087, 604, 2260, 7897, 3129, 616, 1593, 69, 230, 1526, 6349, 6452, 4235, 1752, 4288, 6377, 1229, 395,
                  4326,
                  5845, 5314, 1542, 6483, 2844, 7088, 4702, 3300, 97, 7817, 6804, 471, 3624, 3773, 7057, 2391, 22, 3293,
                  6619, 1933, 6871, 164, 7796, 6744, 1589, 1802, 2880, 7093, 906, 389, 7892, 976, 848, 4076, 7818, 5556,
                  3507, 4740, 4359, 7105, 2938, 683, 4292, 1849, 3121, 5618, 4407, 2883, 7502, 5922, 6130, 301, 4370,
                  7019,
                  3009, 425, 2601, 3592, 790, 2656, 5455, 257, 1500, 3544, 818, 2221, 3313, 3426, 5915, 7155, 3110,
                  4425,
                  5255, 2140, 5632, 614, 1663, 1787, 4023, 1734, 4528, 3318, 4099, 5383, 3999, 722, 3866, 1401, 1299,
                  2926,
                  1360, 1916, 3259, 2420, 1409, 2817, 5961, 782, 1636, 4168, 1344, 4327, 7780, 7335, 3017, 6582, 4623,
                  7198,
                  2499, 2139, 3821, 4822, 2552, 4904, 4328, 6666, 4389, 3687, 1014, 7829, 4802, 5149, 4199, 1866, 1992,
                  2893,
                  6957, 3099, 1212, 672, 4616, 758, 6421, 2281, 6528, 3148, 4197, 1317, 4258, 1407, 6618, 2562, 4448,
                  6137,
                  6151, 1817, 3278, 3982, 5144, 3311, 3453, 1722, 4912, 3641, 5560, 2234, 6645, 3084, 4890, 557, 1455,
                  4152,
                  5784, 7221, 3078, 6961, 23, 4281, 6012, 156, 5109, 6984, 6140, 6730, 4965, 7123, 85, 2912, 5192, 1425,
                  1993, 4056, 598]
        return sample
