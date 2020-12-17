from abc import abstractmethod
from pathlib import Path

import flair.nn
from flair.data import Corpus

from .parameter_groups import *


class DownstreamTaskModel(object):
    """
    Parent class for all downstream tasks models like Sequence Tagging or Text Classification.
    Child classes need to overwrite _set_up_model() and train() in which one has to take care
    of handling document and word specific embeddings.
    """

    def __init__(self):
        pass

    @abstractmethod
    def _set_up_model(self, params: dict, label_dictionary) -> flair.nn.Model:
        """
        sets up downstream task model according to given parameters
        :param params: dict containing key-value pairs as "hyperparameter":"value for hyperparameter" (value is not necessarily a string)
        :return: flair.nn.Model
        """
        pass

    @abstractmethod
    def train(self, corpus: Corpus, params: dict, base_path: Path, max_epochs: int, optimization_value: str) -> dict:
        """
        trains a downstream task and returns a dict containing the configuration and its result
        :param corpus: task to optimize over
        :param params: dict containing the hyperparameter for a single training run
        :param base_path: storage path
        :param max_epochs: max iterations for training the model
        :param optimization_value: metric to be optimized
        :return: dict containing result and configuration
        """
        pass

    @staticmethod
    def _make_word_embeddings_from_attributes(word_embedding_attributes: list) -> list:
        """
        Instantiate word embeddings object during runtime, since only class reference is kept in parameter storage object.
        Otherwise all word embeddings instances during optimization will be kept in memory.
        :param word_embedding_attributes: list containing dicts of word embeddings as "word embedding class":"class parameters"
        :return: instances of word embeddings
        """
        word_embeddings = []

        for idx, embedding in enumerate(word_embedding_attributes):
            WordEmbeddingClass = embedding.get("__class__")
            instance_parameters = {parameter: value for parameter, value in embedding.items() if
                                   parameter != "__class__"}
            word_embeddings.append(WordEmbeddingClass(**instance_parameters))

        return word_embeddings


class TextClassification(DownstreamTaskModel):
    """
    Text Classification downstream task.
    """

    def __init__(self, multi_label: bool = False):
        super().__init__()
        self.multi_label = multi_label

    def _set_up_model(self, params: dict, label_dictionary):
        """
        setup method for text classification downstream tasks, handling document and word embeddings for optimization
        :param params: dict containing the parameters
        :return: a text classifier instance
        """

        # needed since we store a pointer in our configurations list
        params_copy = params.copy()

        document_embedding_name = params_copy['document_embeddings'].__name__

        document_embedding_parameters = self._get_document_embedding_parameters(document_embedding_name, params_copy)

        DocumentEmbeddingClass = params_copy.pop("document_embeddings")

        if "embeddings" in params_copy:
            word_embeddings_attributes = params_copy.pop("embeddings")
            document_embedding_parameters["embeddings"] = self._make_word_embeddings_from_attributes(
                word_embeddings_attributes)

        DocumentEmbedding = DocumentEmbeddingClass(**document_embedding_parameters)

        text_classifier: TextClassifier = TextClassifier(
            label_dictionary=label_dictionary,
            multi_label=self.multi_label,
            document_embeddings=DocumentEmbedding,
        )

        return text_classifier

    @staticmethod
    def _get_document_embedding_parameters(document_embedding_class: str, params: dict):
        """
        Filter method for document embedding class attributes
        :param document_embedding_class: string defining the Document Embedding
        :param params: dict containing the parameters for current training run. will be filtered to use only
        the respective document embedding parameters
        :return: dict containing only the parameters for the document embedding class
        """

        if document_embedding_class == "DocumentEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_EMBEDDING_PARAMETERS
            }

        elif document_embedding_class == "DocumentRNNEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_RNN_EMBEDDING_PARAMETERS
            }

        elif document_embedding_class == "DocumentPoolEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_POOL_EMBEDDING_PARAMETERS
            }

        elif document_embedding_class == "DocumentLMEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_POOL_EMBEDDING_PARAMETERS
            }

        elif document_embedding_class == "TransformerDocumentEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_TRANSFORMER_EMBEDDING_PARAMETERS
            }

        elif document_embedding_class == "SentenceTransformerDocumentEmbeddings":
            embedding_params = {
                key: params[key] for key, value in params.items() if key in DOCUMENT_TRANSFORMER_EMBEDDING_PARAMETERS
            }

        else:
            raise Exception("Please provide a flair document embedding class")

        return embedding_params

    def train(self, corpus: Corpus, params: dict, base_path: Path, max_epochs: int, optimization_value: str):
        """
        trains a text classification model
        :param params: dict containing the parameters
        :return: dict containing result and configuration
        """

        corpus = corpus

        label_dict = corpus.make_label_dictionary()

        for sent in corpus.get_all_sentences():
            sent.clear_embeddings()

        model = self._set_up_model(params, label_dict)

        training_parameters = {
            key: params[key] for key, value in params.items() if key in TRAINING_PARAMETERS
        }

        model_trainer_parameters = {
            key: params[key] for key, value in params.items() if key in MODEL_TRAINER_PARAMETERS and key != 'model'
        }

        trainer: ModelTrainer = ModelTrainer(
            model, corpus, **model_trainer_parameters
        )

        path = base_path

        results = trainer.train(
            path,
            max_epochs=max_epochs,
            param_selection_mode=True,
            **training_parameters
        )

        if optimization_value == "score":
            result = results['test_score']
        else:
            result = results['dev_loss_history'][-1]

        return {'result': result, 'params': params}


class SequenceTagging(DownstreamTaskModel):
    """
    Instantiates a sequence tagger object
    """

    def __init__(self, tag_type: str):
        super().__init__()
        self.tag_type = tag_type

    def train(self, corpus: Corpus, params: dict, base_path: Path, max_epochs: int, optimization_value: str):
        """
        trains a sequence tagger model
        :param params: dict containing the parameters
        :return: dict containing result and configuration
        """

        corpus = corpus

        tag_dictionary = corpus.make_tag_dictionary(self.tag_type)

        tagger = self._set_up_model(params=params, tag_dictionary=tag_dictionary)

        training_params = {
            key: params[key] for key, value in params.items() if key in TRAINING_PARAMETERS
        }
        model_trainer_parameters = {
            key: params[key] for key, value in params.items() if key in MODEL_TRAINER_PARAMETERS and key != 'model'
        }

        trainer: ModelTrainer = ModelTrainer(
            tagger, corpus, **model_trainer_parameters
        )

        path = base_path

        results = trainer.train(path,
                                max_epochs=max_epochs,
                                **training_params)

        if optimization_value == "score":
            result = results['test_score']
        else:
            result = results['dev_loss_history'][-1]

        return {'result': result, 'params': params}

    def _set_up_model(self, params: dict, tag_dictionary):
        """
        setup method for sequence classification downstream tasks, handling word embeddings for optimization.
        :param params: dict containing the parameters
        :return: a sequence tagger instance
        """

        sequence_tagger_params = {
            key: params[key] for key in params if key in SEQUENCE_TAGGER_PARAMETERS
        }

        if "embeddings" in params:
            word_embeddings_attributes = params.pop("embeddings")
            word_embeddings = self._make_word_embeddings_from_attributes(word_embeddings_attributes)

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=word_embeddings)

        sequence_tagger_params['embeddings'] = embeddings

        tagger: SequenceTagger = SequenceTagger(
            tag_dictionary=tag_dictionary,
            tag_type=self.tag_type,
            **sequence_tagger_params,
        )

        return tagger
