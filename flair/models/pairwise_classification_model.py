from typing import Union, List

import torch

import flair.embeddings
import flair.nn
from flair.data import Label, DataPoint, Sentence, DataPair


class TextPairClassifier(flair.nn.DefaultClassifier):
    """
    Text Pair Classification Model for tasks such as Recognizing Textual Entailment, build upon TextClassifier.
    The model takes document embeddings and puts resulting text representation(s) into a linear layer to get the
    actual class label. We provide two ways to embed the DataPairs: Either by embedding both DataPoints
    and concatenating the resulting vectors ("embed_separately=True") or by concatenating the DataPoints and embedding
    the resulting vector ("embed_separately=False").
    """

    def __init__(
            self,
            document_embeddings: flair.embeddings.DocumentEmbeddings,
            label_type: str,
            embed_separately: bool = False,
            **classifierargs,
    ):
        """
        Initializes a TextClassifier
        :param document_embeddings: embeddings used to embed each data point
        :param label_dictionary: dictionary of labels you want to predict
        :param multi_label: auto-detected by default, but you can set this to True to force multi-label prediction
        or False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        """
        super().__init__(**classifierargs)

        self.document_embeddings: flair.embeddings.DocumentEmbeddings = document_embeddings

        self._label_type = label_type

        self.embed_separately = embed_separately

        # if embed_separately == True the linear layer needs twice the length of the embeddings as input size
        # since we concatenate the embeddings of the two DataPoints in the DataPairs
        if self.embed_separately:
            self.decoder = torch.nn.Linear(
                2 * self.document_embeddings.embedding_length, len(self.label_dictionary)
            ).to(flair.device)

            torch.nn.init.xavier_uniform_(self.decoder.weight)

        else:
            # representation for both sentences
            self.decoder = torch.nn.Linear(self.document_embeddings.embedding_length, len(self.label_dictionary))

            # set separator to concatenate two sentences
            self.sep = ' '
            if isinstance(self.document_embeddings, flair.embeddings.document.TransformerDocumentEmbeddings):
                if self.document_embeddings.tokenizer.sep_token:
                    self.sep = ' ' + str(self.document_embeddings.tokenizer.sep_token) + ' '
                else:
                    self.sep = ' [SEP] '

        torch.nn.init.xavier_uniform_(self.decoder.weight)

        # auto-spawn on GPU if available
        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    def forward_pass(self,
                     datapairs: Union[List[DataPoint], DataPoint],
                     return_label_candidates: bool = False,
                     ):

        if isinstance(datapairs, DataPair):
            datapairs = [datapairs]

        embedding_names = self.document_embeddings.get_names()

        if self.embed_separately:  # embed both sentences seperately, concatenate the resulting vectors
            first_elements = [pair.first for pair in datapairs]
            second_elements = [pair.second for pair in datapairs]

            self.document_embeddings.embed(first_elements)
            self.document_embeddings.embed(second_elements)

            text_embedding_list = [
                torch.cat([a.get_embedding(embedding_names), b.get_embedding(embedding_names)], 0).unsqueeze(0)
                for (a, b) in zip(first_elements, second_elements)
            ]

        else:  # concatenate the sentences and embed together
            concatenated_sentences = [
                Sentence(
                    pair.first.to_tokenized_string() + self.sep + pair.second.to_tokenized_string(),
                    use_tokenizer=False
                )
                for pair in datapairs]

            self.document_embeddings.embed(concatenated_sentences)

            text_embedding_list = [
                sentence.get_embedding(embedding_names).unsqueeze(0) for sentence in concatenated_sentences
            ]

        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)

        # linear layer
        scores = self.decoder(text_embedding_tensor)

        labels = []
        for pair in datapairs:
            labels.append([label.value for label in pair.get_labels(self.label_type)])

        # minimal return is scores and labels
        return_tuple = (scores, labels)

        if return_label_candidates:
            label_candidates = [Label(value=None) for pair in datapairs]
            return_tuple += (datapairs, label_candidates)

        return return_tuple

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "document_embeddings": self.document_embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "multi_label": self.multi_label,
            "multi_label_threshold": self.multi_label_threshold,
            "loss_weights": self.loss_weights,
            "embed_separately": self.embed_separately,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        model = TextPairClassifier(
            document_embeddings=state["document_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            multi_label=state["multi_label"],
            multi_label_threshold=0.5 if "multi_label_threshold" not in state.keys() else state["multi_label_threshold"],
            loss_weights=state["loss_weights"],
            embed_separately=state["embed_separately"],
        )
        model.load_state_dict(state["state_dict"])
        return model
