from typing import List, Dict, Union

import torch

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

    def save(self, savefile):
        import pickle
        with open(savefile, 'wb') as f:
            mappings = {
                'idx2item': self.idx2item,
                'item2idx': self.item2idx
            }
            pickle.dump(mappings, f)

    @classmethod
    def load_from_file(cls, filename: str):
        import pickle
        dictionary: Dictionary = Dictionary()
        with open(filename, 'rb') as f:
            mappings = pickle.load(f, encoding='latin1')
            idx2item = mappings['idx2item']
            item2idx = mappings['item2idx']
            dictionary.item2idx = item2idx
            dictionary.idx2item = idx2item
        return dictionary

    @classmethod
    def load(cls, name: str):
        from flair.file_utils import cached_path
        if name == 'chars' or name == 'common-chars':
            base_path = 'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models/common_characters'
            char_dict = cached_path(base_path, cache_dir='datasets')
            return Dictionary.load_from_file(char_dict)

        return Dictionary.load_from_file(name)


class Label:
    """
    This class represents a label of a sentence. Each label has a name and optional a confidence value. The confidence
    value needs to be between 0.0 and 1.0. Default value for the confidence is 1.0.
    """

    def __init__(self, name: str, confidence: float = 1.0):
        self.name = name
        self.confidence = confidence

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if not name:
            raise ValueError('Incorrect label name provided. Label name needs to be set.')
        else:
            self._name = name

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, confidence):
        if 0.0 <= confidence <= 1.0:
            self._confidence = confidence
        else:
            self._confidence = 0.0

    def __str__(self):
        return "{} ({})".format(self._name, self._confidence)

    def __repr__(self):
        return "{} ({})".format(self._name, self._confidence)


class Token:
    """
    This class represents one word in a tokenized sentence. Each token may have any number of tags. It may also point
    to its head in a dependency tree.
    """

    def __init__(self,
                 text: str,
                 idx: int = None,
                 head_id: int = None,
                 whitespace_after: bool = True,
                 ):
        self.text: str = text
        self.idx: int = idx
        self.head_id: int = head_id
        self.whitespace_after: bool = whitespace_after

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

    def __repr__(self) -> str:
        return 'Token: %d %s' % (self.idx, self.text)

    def set_embedding(self, name: str, vector: torch.autograd.Variable):
        self._embeddings[name] = vector.cpu()

    def clear_embeddings(self):
        self._embeddings: Dict = {}

    def get_embedding(self) -> torch.autograd.Variable:

        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            embeddings.append(self._embeddings[embed])

        if embeddings:
            return torch.cat(embeddings, dim=0)

        return torch.FloatTensor()

    @property
    def embedding(self):
        return self.get_embedding()


class Sentence:
    def __init__(self, text: str = None, use_tokenizer: bool = False, labels: Union[List[Label], List[str]] = None):

        super(Sentence, self).__init__()

        self.tokens: List[Token] = []

        self.labels: List[Label] = []
        if labels is not None: self.add_labels(labels)

        self._embeddings: Dict = {}

        # if text is passed, instantiate sentence with tokens (words)
        if text is not None:

            # tokenize the text first if option selected
            if use_tokenizer:

                # use segtok for tokenization
                tokens = []
                sentences = split_single(text)
                for sentence in sentences:
                    contractions = split_contractions(word_tokenizer(sentence))
                    tokens.extend(contractions)

                # determine offsets for whitespace_after field
                index = text.index
                running_offset = 0
                last_word_offset = -1
                last_token = None
                for word in tokens:
                    token = Token(word)
                    self.add_token(token)
                    try:
                        word_offset = index(word, running_offset)
                    except:
                        word_offset = last_word_offset + 1
                    if word_offset - 1 == last_word_offset and last_token is not None:
                        last_token.whitespace_after = False
                    word_len = len(word)
                    running_offset = word_offset + word_len
                    last_word_offset = running_offset - 1
                    last_token = token

            # otherwise assumes whitespace tokenized text
            else:
                # add each word in tokenized string as Token object to Sentence
                for word in text.split(' '):
                    if word:
                        token = Token(word)
                        self.add_token(token)

    def _infer_space_after(self):
        """
        Heuristics in case you wish to infer whitespace_after values for tokenized text. This is useful for some old NLP
        tasks (such as CoNLL-03 and CoNLL-2000) that provide only tokenized data with no info of original whitespacing.
        :return:
        """
        last_token = None
        quote_count: int = 0
        # infer whitespace after field

        for token in self.tokens:
            if token.text == '"':
                quote_count += 1
                if quote_count % 2 != 0:
                    token.whitespace_after = False
                elif last_token is not None:
                    last_token.whitespace_after = False

            if last_token is not None:

                if token.text in ['.', ':', ',', ';', ')', 'n\'t', '!', '?']:
                    last_token.whitespace_after = False

                if token.text.startswith('\''):
                    last_token.whitespace_after = False

            if token.text in ['(']:
                token.whitespace_after = False

            last_token = token
        return self

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def add_label(self, label: Union[Label, str]):
        if type(label) is Label:
            self.labels.append(label)

        elif type(label) is str:
            self.labels.append(Label(label))

    def add_labels(self, labels: Union[List[Label], List[str]]):
        for label in labels:
            self.add_label(label)

    def get_label_names(self) -> List[str]:
        return [label.name for label in self.labels]

    def get_token(self, token_id: int) -> Token:
        for token in self.tokens:
            if token.idx == token_id:
                return token

    def add_token(self, token: Token):
        self.tokens.append(token)

        # set token idx if not set
        token.sentence = self
        if token.idx is None:
            token.idx = len(self.tokens)

    def set_embedding(self, name: str, vector):
        self._embeddings[name] = vector.cpu()

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

        if embeddings:
            return torch.cat(embeddings, dim=0)

        return torch.FloatTensor()

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

    def __len__(self) -> int:
        return len(self.tokens)

    def to_tokenized_string(self) -> str:
        return ' '.join([t.text for t in self.tokens])

    def to_plain_string(self):
        plain = ''
        for token in self.tokens:
            plain += token.text
            if token.whitespace_after: plain += ' '
        return plain.rstrip()


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

        labels = set(self._get_all_label_names())

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

    def _get_most_common_tokens(self, max_tokens, min_freq) -> List[str]:
        tokens_and_frequencies = Counter(self._get_all_tokens())
        tokens_and_frequencies = tokens_and_frequencies.most_common()

        tokens = []
        for token, freq in tokens_and_frequencies:
            if (min_freq != -1 and freq < min_freq) or (max_tokens != -1 and len(tokens) == max_tokens):
                break
            tokens.append(token)
        return tokens

    def _get_all_label_names(self) -> List[str]:
        return [label.name for sent in self.train for label in sent.labels]

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

    @staticmethod
    def _print_statistics_for(sentences, name):
        if len(sentences) == 0:
            return

        classes_to_count = TaggedCorpus._get_classes_to_count(sentences)
        tokens_per_sentence = TaggedCorpus._get_tokens_per_sentence(sentences)

        print(name)
        print("total size: " + str(len(sentences)))
        for l, c in classes_to_count.items():
            print("size of class {}: {}".format(l, c))
        print("total # of tokens: " + str(sum(tokens_per_sentence)))
        print("min # of tokens: " + str(min(tokens_per_sentence)))
        print("max # of tokens: " + str(max(tokens_per_sentence)))
        print("avg # of tokens: " + str(sum(tokens_per_sentence) / len(sentences)))

    @staticmethod
    def _get_tokens_per_sentence(sentences):
        return list(map(lambda x: len(x.tokens), sentences))

    @staticmethod
    def _get_classes_to_count(sentences):
        classes_to_count = defaultdict(lambda: 0)
        for sent in sentences:
            for label in sent.labels:
                classes_to_count[label.name] += 1
        return classes_to_count

    def __str__(self) -> str:
        return 'TaggedCorpus: %d train + %d dev + %d test sentences' % (len(self.train), len(self.dev), len(self.test))


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
