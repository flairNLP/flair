from typing import List, Dict
import torch
import random
from random import randint
import os
from os import listdir
from os.path import isfile, join
from collections import Counter
from collections import defaultdict

from segtok.segmenter import split_single
from segtok.tokenizer import split_contractions
from segtok.tokenizer import word_tokenizer


class Dictionary:
    """
    This class holds a dictionary that maps strings to IDs, used to generate one-hot encodings of strings.
    """

    def __init__(self, add_unk=True):
        # init dictionaries
        self.item2idx: Dict[str, int] = {}
        self.idx2item: List[str] = []

        # in order to deal with unknown tokens, add <unk>
        if add_unk:
            self.add_item('<unk>')

    def add_item(self, item: str) -> int:
        """
        add string - if already in dictionary returns its ID. if not in dictionary, it will get a new ID.
        :param item: a string for which to assign an id
        :return: ID of string
        """
        item = item.encode('utf-8')
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1
        return self.item2idx[item]

    def get_idx_for_item(self, item: str) -> int:
        """
        returns the ID of the string, otherwise 0
        :param item: string for which ID is requested
        :return: ID of string, otherwise 0
        """
        item = item.encode('utf-8')
        if item in self.item2idx.keys():
            return self.item2idx[item]
        else:
            return 0

    def get_items(self) -> List[str]:
        items = []
        for item in self.idx2item:
            items.append(item.decode('UTF-8'))
        return items

    def __len__(self) -> int:
        return len(self.idx2item)

    def get_item_for_index(self, idx):
        return self.idx2item[idx].decode('UTF-8')


class Token:
    """
    This class represents one word in a tokenized sentence. Each token may have any number of tags. It may also point
    to its head in a dependency tree.
    """

    def __init__(self,
                 text: str,
                 idx: int = None,
                 head_id: int = None
                 ):
        self.text: str = text
        self.idx: int = idx
        self.head_id: int = head_id

        self.sentence: Sentence = None
        self._embeddings: Dict = {}
        self.tags: Dict[str, str] = {}

    def add_tag(self, tag_type: str, tag_value: str):
        self.tags[tag_type] = tag_value

    def get_tag(self, tag_type: str) -> str:
        if tag_type in self.tags: return self.tags[tag_type]
        return ''

    def get_head(self):
        return self.sentence.get_token(self.head_id)

    def __str__(self) -> str:
        return 'Token: %d %s' % (self.idx, self.text)

    def set_embedding(self, name: str, vector: torch.autograd.Variable):
        self._embeddings[name] = vector

    def clear_embeddings(self):
        self._embeddings: Dict = {}

    def get_embedding(self) -> torch.autograd.Variable:

        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            embeddings.append(self._embeddings[embed])

        return torch.cat(embeddings, dim=0)

    @property
    def embedding(self):
        return self.get_embedding()


class Sentence:
    def __init__(self, text: str = None, use_tokenizer: bool = False, labels: List[str] = None):

        super(Sentence, self).__init__()

        self.tokens: List[Token] = []

        self.labels: List[str] = labels

        self._embeddings: Dict = {}

        # optionally, directly instantiate with sentence tokens
        if text is not None:

            # tokenize the text first if option selected, otherwise assumes whitespace tokenized text
            if use_tokenizer:
                sentences = split_single(text)
                tokens = []
                for sentence in sentences:
                    contractions = split_contractions(word_tokenizer(sentence))
                    tokens.extend(contractions)

                text = ' '.join(tokens)

            # add each word in tokenized string as Token object to Sentence
            for word in text.split(' '):
                self.add_token(Token(word))

    def __getitem__(self, token_id: int) -> Token:
        return self.get_token(token_id)

    def __iter__(self):
        return iter(self.tokens)

    def get_token(self, token_id: int) -> Token:
        for token in self.tokens:
            if token.idx == token_id: return token

    def add_token(self, token: Token):
        self.tokens.append(token)

        # set token idx if not set
        token.sentence = self
        if token.idx is None:
            token.idx = len(self.tokens)

    def set_embedding(self, name: str, vector):
        self._embeddings[name] = vector

    def clear_embeddings(self, also_clear_word_embeddings: bool = True):

        self._embeddings: Dict = {}

        if also_clear_word_embeddings:
            for token in self:
                token.clear_embeddings()

    def cpu_embeddings(self):
        for name, vector in self._embeddings.items():
            self._embeddings[name] = vector.cpu()

    def get_embedding(self) -> torch.autograd.Variable:
        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            embedding = self._embeddings[embed]
            embeddings.append(embedding)

        return torch.cat(embeddings, dim=0)

    @property
    def embedding(self):
        return self.get_embedding()

    def to_tagged_string(self) -> str:

        list = []
        for token in self.tokens:
            list.append(token.text)

            tags = []
            for tag_type in token.tags.keys():

                if token.get_tag(tag_type) == '' or token.get_tag(tag_type) == 'O': continue
                tags.append(token.get_tag(tag_type))
            all_tags = '<' + '/'.join(tags) + '>'
            if all_tags != '<>':
                list.append(all_tags)
        return ' '.join(list)

    # def to_tag_string(self, tag_type: str = 'tag') -> str:
    #
    #     list = []
    #     for token in self.tokens:
    #         list.append(token.text)
    #         if token.get_tag(tag_type) == '' or token.get_tag(tag_type) == 'O': continue
    #         list.append('<' + token.get_tag(tag_type) + '>')
    #     return ' '.join(list)
    #
    # def to_ner_string(self) -> str:
    #     list = []
    #     for token in self.tokens:
    #         if token.get_tag('ner') == 'O' or token.get_tag('ner') == '':
    #             list.append(token.text)
    #         else:
    #             list.append(token.text)
    #             list.append('<' + token.get_tag('ner') + '>')
    #     return ' '.join(list)

    def convert_tag_scheme(self, tag_type: str = 'ner', target_scheme: str = 'iob'):

        tags: List[str] = []
        for token in self.tokens:
            token: Token = token
            tags.append(token.get_tag(tag_type))

        if target_scheme == 'iob':
            iob2(tags)

        if target_scheme == 'iobes':
            iob2(tags)
            tags = iob_iobes(tags)

        for index, tag in enumerate(tags):
            self.tokens[index].add_tag(tag_type, tag)

    def __repr__(self):
        return 'Sentence: "' + ' '.join([t.text for t in self.tokens]) + '" - %d Tokens' % len(self)

    def __copy__(self):
        s = Sentence()
        for token in self.tokens:
            nt = Token(token.text)
            for tag_type in token.tags:
                nt.add_tag(tag_type, token.get_tag(tag_type))

            s.add_token(nt)
        return s

    def __str__(self) -> str:
        return 'Sentence: "' + ' '.join([t.text for t in self.tokens]) + '" - %d Tokens' % len(self)

    def to_plain_string(self) -> str:
        return ' '.join([t.text for t in self.tokens])

    def __len__(self) -> int:
        return len(self.tokens)


class TaggedCorpus:
    def __init__(self, train: List[Sentence], dev: List[Sentence], test: List[Sentence]):
        self.train: List[Sentence] = train
        self.dev: List[Sentence] = dev
        self.test: List[Sentence] = test

    def downsample(self, percentage: float = 0.1, only_downsample_train=False):

        self.train = self._downsample_to_proportion(self.train, percentage)
        if not only_downsample_train:
            self.dev = self._downsample_to_proportion(self.dev, percentage)
            self.test = self._downsample_to_proportion(self.test, percentage)

        return self

    def clear_embeddings(self):
        for sentence in self.get_all_sentences():
            for token in sentence.tokens:
                token.clear_embeddings()

    def get_all_sentences(self) -> List[Sentence]:
        all_sentences: List[Sentence] = []
        all_sentences.extend(self.train)
        all_sentences.extend(self.dev)
        all_sentences.extend(self.test)
        return all_sentences

    def make_tag_dictionary(self, tag_type: str) -> Dictionary:

        # Make the tag dictionary
        tag_dictionary: Dictionary = Dictionary()
        tag_dictionary.add_item('O')
        for sentence in self.get_all_sentences():
            for token in sentence.tokens:
                token: Token = token
                tag_dictionary.add_item(token.get_tag(tag_type))
        tag_dictionary.add_item('<START>')
        tag_dictionary.add_item('<STOP>')
        return tag_dictionary

    def make_label_dictionary(self) -> Dictionary:
        """
        Creates a dictionary of all labels assigned to the sentences in the corpus.
        :return: dictionary of labels
        """

        labels = set(self._get_all_labels())

        label_dictionary: Dictionary = Dictionary(add_unk=False)
        for label in labels:
            label_dictionary.add_item(label)

        return label_dictionary

    def make_vocab_dictionary(self, max_tokens=-1, min_freq=1) -> Dictionary:
        """
        Creates a dictionary of all tokens contained in the corpus.
        By defining `max_tokens` you can set the maximum number of tokens that should be contained in the dictionary.
        If there are more than `max_tokens` tokens in the corpus, the most frequent tokens are added first.
        If `min_freq` is set the a value greater than 1 only tokens occurring more than `min_freq` times are considered
        to be added to the dictionary.
        :param max_tokens: the maximum number of tokens that should be added to the dictionary (-1 = take all tokens)
        :param min_freq: a token needs to occur at least `min_freq` times to be added to the dictionary (-1 = there is no limitation)
        :return: dictionary of tokens
        """
        tokens = self._get_most_common_tokens(max_tokens, min_freq)

        vocab_dictionary: Dictionary = Dictionary()
        for token in tokens:
            vocab_dictionary.add_item(token)

        return vocab_dictionary

    def _get_most_common_tokens(self, max_tokens, min_freq) -> List[Token]:
        tokens_and_frequencies = Counter(self._get_all_tokens())
        tokens_and_frequencies = tokens_and_frequencies.most_common()

        tokens = []
        for token, freq in tokens_and_frequencies:
            if freq <= min_freq or len(tokens) == max_tokens:
                break
            tokens.append(token)
        return tokens

    def _get_all_labels(self) -> List[str]:
        return [label for sent in self.train for label in sent.labels]

    def _get_all_tokens(self) -> List[str]:
        tokens = list(map((lambda s: s.tokens), self.train))
        tokens = [token for sublist in tokens for token in sublist]
        return list(map((lambda t: t.text), tokens))

    def _downsample_to_proportion(self, list: List, proportion: float):

        counter = 0.0
        last_counter = None
        downsampled: List = []

        for item in list:
            counter += proportion
            if int(counter) != last_counter:
                downsampled.append(item)
                last_counter = int(counter)
        return downsampled

    def print_statistics(self):
        """
        Print statistics about the class distribution (only labels of sentences are taken into account) and sentence
        sizes.
        """

        self._print_statistics_for(self.train, "TRAIN")
        self._print_statistics_for(self.test, "TEST")
        self._print_statistics_for(self.dev, "DEV")

    def _print_statistics_for(self, dataset, name):
        if len(dataset) == 0:
            return

        classes_to_count = defaultdict(lambda: 0)
        for sent in dataset:
            for label in sent.labels:
                classes_to_count[label] += 1
        tokens_per_doc = list(map(lambda x: len(x.tokens), dataset))

        print(name)
        print("total size: " + str(len(dataset)))
        for l, c in classes_to_count.items():
            print("size of class {}: {}".format(l, c))
        print("total # of tokens: " + str(sum(tokens_per_doc)))
        print("min # of tokens: " + str(min(tokens_per_doc)))
        print("max # of tokens: " + str(max(tokens_per_doc)))
        print("avg # of tokens: " + str(sum(tokens_per_doc) / len(dataset)))

    def __str__(self) -> str:
        return 'TaggedCorpus: %d train + %d dev + %d test sentences' % (len(self.train), len(self.dev), len(self.test))




class CorpusLM(object):
    def __init__(self, path, dictionary: Dictionary, forward: bool = True, character_level: bool = True):
        self.dictionary: Dictionary = dictionary
        self.train_path = os.path.join(path, 'train')
        self.train = None
        self.forward = forward
        self.split_on_char = character_level

        self.train_files = sorted([f for f in listdir(self.train_path) if isfile(join(self.train_path, f))])
        self.current_train_file = None

        if forward:
            self.valid = self.charsplit(os.path.join(path, 'valid.txt'), expand_vocab=False, forward=True,
                                        split_on_char=self.split_on_char)
            self.test = self.charsplit(os.path.join(path, 'test.txt'), expand_vocab=False, forward=True,
                                       split_on_char=self.split_on_char)
        else:
            self.valid = self.charsplit(os.path.join(path, 'valid.txt'), expand_vocab=False, forward=False,
                                        split_on_char=self.split_on_char)
            self.test = self.charsplit(os.path.join(path, 'test.txt'), expand_vocab=False, forward=False,
                                       split_on_char=self.split_on_char)

    def get_next_train_slice(self) -> str:

        if self.current_train_file == None:
            self.current_train_file = self.train_files[0]

        elif len(self.train_files) != 1:

            index = self.train_files.index(self.current_train_file) + 1
            if index > len(self.train_files): index = 0

            self.current_train_file = self.train_files[index]

            self.train = self.charsplit(os.path.join(self.train_path, self.current_train_file), expand_vocab=False,
                                        forward=self.forward, split_on_char=self.split_on_char)

        return self.current_train_file

    def get_random_train_slice(self) -> str:
        train_files = [f for f in listdir(self.train_path) if isfile(join(self.train_path, f))]
        current_train_file = random.choice(train_files)
        self.train = self.charsplit(os.path.join(self.train_path, current_train_file), expand_vocab=False,
                                    forward=self.forward, split_on_char=self.split_on_char)
        return current_train_file

    def charsplit(self, path: str, expand_vocab=False, forward=True, split_on_char=True) -> torch.LongTensor:

        """Tokenizes a text file on characted basis."""
        assert os.path.exists(path)

        #
        with open(path, 'r', encoding="utf-8") as f:
            tokens = 0
            for line in f:

                if split_on_char:
                    chars = list(line)
                else:
                    chars = line.split()

                # print(chars)
                tokens += len(chars)

                # Add chars to the dictionary
                if expand_vocab:
                    for char in chars:
                        self.dictionary.add_item(char)

        if forward:
            # charsplit file content
            with open(path, 'r', encoding="utf-8") as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    line = self.random_casechange(line)

                    if split_on_char:
                        chars = list(line)
                    else:
                        chars = line.split()

                    for char in chars:
                        if token >= tokens: break
                        ids[token] = self.dictionary.get_idx_for_item(char)
                        token += 1
        else:
            # charsplit file content
            with open(path, 'r', encoding="utf-8") as f:
                ids = torch.LongTensor(tokens)
                token = tokens - 1
                for line in f:
                    line = self.random_casechange(line)

                    if split_on_char:
                        chars = list(line)
                    else:
                        chars = line.split()

                    for char in chars:
                        if token >= tokens: break
                        ids[token] = self.dictionary.get_idx_for_item(char)
                        token -= 1

        return ids

    def random_casechange(self, line: str) -> str:
        no = randint(0, 99)
        if no is 0:
            line = line.lower()
        if no is 1:
            line = line.upper()
        return line

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                            tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                            tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags
