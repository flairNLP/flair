import os
import re
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, default_data_collator
from huggingface_hub import hf_hub_url, cached_download
from string import punctuation
import flair
from flair.data import Sentence, SpanLabel
from typing import List, Union
from pathlib import Path

log = logging.getLogger("flair")


class SparseEncoder(object):
    def __init__(self, use_cuda=False):
        self.encoder = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
        self.use_cuda = use_cuda

    def fit(self, train_corpus):
        self.encoder.fit(train_corpus)
        return self

    def transform(self, mentions):
        vec = self.encoder.transform(mentions).toarray()
        vec = torch.FloatTensor(vec)  # return torch float tensor
        if self.use_cuda:
            vec = vec.cuda()
        return vec

    def cuda(self):
        self.use_cuda = True

        return self

    def cpu(self):
        self.use_cuda = False
        return self

    def __call__(self, mentions):
        return self.transform(mentions)

    def vocab(self):
        return self.encoder.vocabulary_

    def save_encoder(self, path):
        with open(path, "wb") as fout:
            pickle.dump(self.encoder, fout)
            log.info("Sparse encoder saved in {}".format(path))

    def load_encoder(self, path):
        with open(path, "rb") as fin:
            self.encoder = pickle.load(fin)
            log.info("Sparse encoder loaded from {}".format(path))

        return self


class TextPreprocess:
    """
    Text Preprocess module
    Support lowercase, removing punctuation, typo correction
    """

    def __init__(
        self,
        lowercase=True,
        remove_punctuation=True,
        ignore_punctuations="",
        typo_path=None,
    ):
        """
        :param typo_path str: path of known typo dictionary
        """
        self.lowercase = lowercase
        self.typo_path = typo_path
        self.rmv_puncts = remove_punctuation
        self.punctuation = punctuation
        for ig_punc in ignore_punctuations:
            self.punctuation = self.punctuation.replace(ig_punc, "")
        self.rmv_puncts_regex = re.compile(
            r"[\s{}]+".format(re.escape(self.punctuation))
        )

        if typo_path:
            self.typo2correction = self.load_typo2correction(typo_path)
        else:
            self.typo2correction = {}

    def load_typo2correction(self, typo_path):
        typo2correction = {}
        with open(typo_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                s = line.strip()
                tokens = s.split("||")
                value = "" if len(tokens) == 1 else tokens[1]
                typo2correction[tokens[0]] = value

        return typo2correction

    def remove_punctuation(self, phrase):
        phrase = self.rmv_puncts_regex.split(phrase)
        phrase = " ".join(phrase).strip()

        return phrase

    def correct_spelling(self, phrase):
        phrase_tokens = phrase.split()
        phrase = ""

        for phrase_token in phrase_tokens:
            if phrase_token in self.typo2correction.keys():
                phrase_token = self.typo2correction[phrase_token]
            phrase += phrase_token + " "

        phrase = phrase.strip()
        return phrase

    def run(self, text):
        if self.lowercase:
            text = text.lower()

        if self.typo_path:
            text = self.correct_spelling(text)

        if self.rmv_puncts:
            text = self.remove_punctuation(text)

        text = text.strip()

        return text


class NamesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        # This gave a warning
        # return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class DictionaryDataset:
    """
    A class used to load dictionary data
    """

    def __init__(self, dictionary_path):
        """
        :param dictionary_path str: The path of the dictionary
        """
        log.info("Loading Dictionary from {}".format(dictionary_path))
        self.data = self.load_data(dictionary_path)

    def load_data(self, dictionary_path):
        data = []
        with open(dictionary_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line == "":
                    continue
                cui, name = line.split("||")
                data.append((name, cui))

        data = np.array(data)
        return data


class BioSyn(object):
    """
    Wrapper class for dense encoder and sparse encoder
    """

    def __init__(self, max_length, use_cuda, initial_sparse_weight=None):
        self.max_length = max_length
        self.use_cuda = use_cuda

        self.tokenizer = None
        self.encoder = None
        self.sparse_encoder = None
        self.sparse_weight = None

        if initial_sparse_weight != None:
            self.sparse_weight = self.init_sparse_weight(initial_sparse_weight)

    def init_sparse_weight(self, initial_sparse_weight):
        """
        :param initial_sparse_weight float: initial sparse weight
        """
        if self.use_cuda:
            self.sparse_weight = nn.Parameter(torch.empty(1).cuda())
        else:
            self.sparse_weight = nn.Parameter(torch.empty(1))
        self.sparse_weight.data.fill_(initial_sparse_weight)  # init sparse_weight

        return self.sparse_weight

    def init_sparse_encoder(self, corpus):
        self.sparse_encoder = SparseEncoder().fit(corpus)

        return self.sparse_encoder

    def get_dense_encoder(self):
        assert self.encoder is not None

        return self.encoder

    def get_dense_tokenizer(self):
        assert self.tokenizer is not None

        return self.tokenizer

    def get_sparse_encoder(self):
        assert self.sparse_encoder is not None

        return self.sparse_encoder

    def get_sparse_weight(self):
        assert self.sparse_weight is not None

        return self.sparse_weight

    def save_model(self, path):
        # save dense encoder
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # save sparse encoder
        sparse_encoder_path = os.path.join(path, "sparse_encoder.pk")
        self.sparse_encoder.save_encoder(path=sparse_encoder_path)

        sparse_weight_file = os.path.join(path, "sparse_weight.pt")
        torch.save(self.sparse_weight, sparse_weight_file)
        log.info("Sparse weight saved in {}".format(sparse_weight_file))

    def load_model(self, model_name_or_path):
        self.load_dense_encoder(model_name_or_path)
        self.load_sparse_encoder(model_name_or_path)
        self.load_sparse_weight(model_name_or_path)

        return self

    def load_dense_encoder(self, model_name_or_path):
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.use_cuda:
            self.encoder = self.encoder.to("cuda")

        return self.encoder, self.tokenizer

    def load_sparse_encoder(self, model_name_or_path):
        sparse_encoder_path = os.path.join(model_name_or_path, "sparse_encoder.pk")
        # check file exists
        if not os.path.isfile(sparse_encoder_path):
            # download from huggingface hub and cache it
            sparse_encoder_url = hf_hub_url(
                model_name_or_path, filename="sparse_encoder.pk"
            )
            sparse_encoder_path = cached_download(
                url=sparse_encoder_url,
                cache_dir=flair.cache_root / "models" / model_name_or_path)

        self.sparse_encoder = SparseEncoder().load_encoder(path=sparse_encoder_path)

        return self.sparse_encoder

    def load_sparse_weight(self, model_name_or_path):
        sparse_weight_path = os.path.join(model_name_or_path, "sparse_weight.pt")
        # check file exists
        if not os.path.isfile(sparse_weight_path):
            # download from huggingface hub and cache it
            sparse_weight_url = hf_hub_url(
                model_name_or_path, filename="sparse_weight.pt"
            )
            sparse_weight_path = cached_download(
                url=sparse_weight_url,
                cache_dir=flair.cache_root / "models" / model_name_or_path
            )

        self.sparse_weight = torch.load(sparse_weight_path)

        return self.sparse_weight

    def get_score_matrix(self, query_embeds, dict_embeds):
        """
        Return score matrix
        :param query_embeds np.array: 2d numpy array of query embeddings
        :param dict_embeds np.array: 2d numpy array of query embeddings
        :returns score_matrix np.array: 2d numpy array of scores
        """
        score_matrix = np.matmul(query_embeds, dict_embeds.T)

        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk indxes (descending order)
        :param score_matrix np.array: 2d numpy array of scores
        :param topk int: The number of candidates
        :returns topk_idxs np.array: 2d numpy array of scores [# of query , # of dict]
        """

        def indexing_2d(arr, cols):
            rows = np.repeat(
                np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1
            )
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix, -topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix)
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def embed_sparse(self, names, show_progress=False):
        """
        Embedding data into sparse representations
        :param names np.array: An array of names
        :returns sparse_embeds np.array: A list of sparse embeddings
        """
        batch_size = 1024
        sparse_embeds = []

        if show_progress:
            iterations = tqdm(range(0, len(names), batch_size))
        else:
            iterations = range(0, len(names), batch_size)

        for start in iterations:
            end = min(start + batch_size, len(names))
            batch = names[start:end]
            batch_sparse_embeds = self.sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def embed_dense(self, names, show_progress=False):
        """
        Embedding data into dense representations
        :param names np.array or list: An array of names
        :returns dense_embeds list: A list of dense embeddings
        """
        self.encoder.eval()  # prevent dropout

        batch_size = 1024
        dense_embeds = []

        if isinstance(names, np.ndarray):
            names = names.tolist()
        name_encodings = self.tokenizer(
            names,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        if self.use_cuda:
            name_encodings = name_encodings.to("cuda")
        name_dataset = NamesDataset(name_encodings)
        name_dataloader = DataLoader(
            name_dataset,
            shuffle=False,
            collate_fn=default_data_collator,
            batch_size=batch_size,
        )

        with torch.no_grad():
            for batch in tqdm(
                name_dataloader, disable=not show_progress, desc="embedding dictionary"
            ):
                outputs = self.encoder(**batch)
                batch_dense_embeds = (
                    outputs[0][:, 0].cpu().detach().numpy()
                )  # [CLS] representations
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)

        return dense_embeds


class HunNen(object):
    """
    Biomedical Entity Linker for HunFlair
    Can predict top k entities on sentences annotated with biomedical entity mentions using BioSyn.
    """

    def __init__(self, model, dictionary: DictionaryDataset, dict_sparse_embeds, dict_dense_embeds):
        """
        Initalize HunNen class, called by classmethod load
        :param model: BioSyn object containing the dense and sparse encoders
        :param dictionary: numpy array containing all concept names and their cui
        :param dict_sparse_embeds: sparse embeddings of dictionary
        :param dict_dense_embeds: dense embeddings of dictionary
        """
        super().__init__()
        self.biosyn = model
        self.dictionary = dictionary
        self.dict_sparse_embeds = dict_sparse_embeds
        self.dict_dense_embeds = dict_dense_embeds
        
    @classmethod
    def load(cls, model_name, dictionary_path:  Union[str, Path]):
        """
        Load a model for biomedical named entity normalization using BioSyn on sentences annotated with 
        biomedical entity mentions
        :param model_name: Name of pretrained model to use. Currently possible values for pretrained models are: 
        sapbert-bc5cdr-disease, sapbert-ncbi-disease, sapbert-bc5cdr-chemical, biobert-bc5cdr-disease, 
        biobert-ncbi-disease, biobert-bc5cdr-chemical
        :param dictionary_path: Path to a file with each line in the format: cui||name, with one line for each
        name of a concept
        """
        # load biosyn
        biosyn = BioSyn(max_length=25, use_cuda=torch.cuda.is_available())

        # modify name if it's a huggingface model
        if model_name in ["sapbert-bc5cdr-disease", "sapbert-ncbi-disease", "sapbert-bc5cdr-chemical", 
        "biobert-bc5cdr-disease", "biobert-ncbi-disease", "biobert-bc5cdr-chemical"]:
            model_name = "dmis-lab/biosyn-" + model_name

        biosyn.load_model(model_name_or_path=model_name)
        

        # cache or load dictionary
        dictionary, dict_sparse_embeds, dict_dense_embeds = cls._cache_or_load_dictionary(
            biosyn, model_name, str(dictionary_path)
        )

        return cls(biosyn, dictionary, dict_sparse_embeds, dict_dense_embeds);

    def predict(self, sentences: Union[List[Sentence], Sentence], entity_type, topk = 10):
        """
        On one or more senteces, predict the cui on all named entites annotated with a tag of type entity_type. 
        Annotates the top k predictions.
        :param sentences: one or more sentences to run the predictions on
        :param entity_type: only entities with this tag will be annotated
        :param topk: number of predicted cui candidates to add to annotation
        """
        # make sure its a list of sentences
        if not isinstance(sentences, list):
            sentences = [sentences]
        
        for sentence in sentences:
            for entity in sentence.get_labels(entity_type):
                # preprocess mention
                mention = TextPreprocess().run(entity.span.text)

                # embed mention
                mention_sparse_embeds = self.biosyn.embed_sparse(names=[mention])
                mention_dense_embeds = self.biosyn.embed_dense(names=[mention])

                # calcuate score matrix and get top 5
                sparse_score_matrix = self.biosyn.get_score_matrix(
                    query_embeds=mention_sparse_embeds, dict_embeds=self.dict_sparse_embeds
                )
                dense_score_matrix = self.biosyn.get_score_matrix(
                    query_embeds=mention_dense_embeds, dict_embeds=self.dict_dense_embeds
                )
                sparse_weight = self.biosyn.get_sparse_weight().item()
                hybrid_score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
                hybrid_candidate_idxs = self.biosyn.retrieve_candidate(
                    score_matrix=hybrid_score_matrix, topk=topk
                )

                # get predictions from dictionary
                predictions = self.dictionary[hybrid_candidate_idxs].squeeze(0)

                for prediction in predictions:
                    predicted_name = prediction[0]
                    predicted_id = prediction[1]
                    sentence.add_complex_label(typename="entity_type" + "_nen", 
                        label=SpanLabel(span=entity.span, value= f"{predicted_name} {predicted_id}"))


    @staticmethod
    def _cache_or_load_dictionary(biosyn, model_name_or_path, dictionary_path):
        dictionary_name = os.path.splitext(os.path.basename(dictionary_path))[0]

        cache_folder = os.path.join(flair.cache_root, "datasets")
        cached_dictionary_path = os.path.join(
            cache_folder, f"cached_{model_name_or_path.split('/')[-1]}_{dictionary_name}.pk"
        )

        # If exist, load the cached dictionary
        if os.path.exists(cached_dictionary_path):
            with open(cached_dictionary_path, "rb") as fin:
                cached_dictionary = pickle.load(fin)
            log.info("Loaded dictionary from cached file {}".format(cached_dictionary_path))

            dictionary, dict_sparse_embeds, dict_dense_embeds = (
                cached_dictionary["dictionary"],
                cached_dictionary["dict_sparse_embeds"],
                cached_dictionary["dict_dense_embeds"],
            )

        else:
            dictionary = DictionaryDataset(dictionary_path=dictionary_path).data
            dictionary_names = dictionary[:, 0]
            dict_sparse_embeds = biosyn.embed_sparse(
                names=dictionary_names, show_progress=True
            )
            dict_dense_embeds = biosyn.embed_dense(
                names=dictionary_names, show_progress=True
            )
            cached_dictionary = {
                "dictionary": dictionary,
                "dict_sparse_embeds": dict_sparse_embeds,
                "dict_dense_embeds": dict_dense_embeds,
            }

            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)
            with open(cached_dictionary_path, "wb") as fin:
                pickle.dump(cached_dictionary, fin)
            print("Saving dictionary into cached file {}".format(cached_dictionary_path))

        return dictionary, dict_sparse_embeds, dict_dense_embeds




