from typing import List

import torch

import flair.embeddings
import flair.nn
from flair.data import Sentence, TextPair, DT2


class TextPairClassifier(flair.nn.DefaultClassifier[TextPair, TextPair]):
    """
    Text Pair Classification Model for tasks such as Recognizing Textual Entailment, build upon TextClassifier.
    The model takes document embeddings and puts resulting text representation(s) into a linear layer to get the
    actual class label. We provide two ways to embed the DataPairs: Either by embedding both DataPoints
    and concatenating the resulting vectors ("embed_separately=True") or by concatenating the DataPoints and embedding
    the resulting vector ("embed_separately=False").
    """

    def __init__(
        self,
        embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        embed_separately: bool = False,
        **classifierargs,
    ):
        """
        Initializes a TextClassifier
        :param embeddings: embeddings used to embed each data point
        :param label_dictionary: dictionary of labels you want to predict
        :param multi_label: auto-detected by default, but you can set this to True to force multi-label prediction
        or False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        """
        super().__init__(
            **classifierargs,
            embeddings=embeddings,
            final_embedding_size=2 * embeddings.embedding_length if embed_separately else embeddings.embedding_length,
        )

        self._label_type = label_type

        self.embed_separately = embed_separately

        if not self.embed_separately:
            # set separator to concatenate two sentences
            self.sep = " "
            if isinstance(
                self.embeddings,
                flair.embeddings.document.TransformerDocumentEmbeddings,
            ):
                if self.embeddings.tokenizer.sep_token:
                    self.sep = " " + str(self.embeddings.tokenizer.sep_token) + " "
                else:
                    self.sep = " [SEP] "

        # auto-spawn on GPU if available
        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    def _get_data_points_from_sentence(self, sentence: TextPair) -> List[TextPair]:
        return [sentence]

    def _get_embedding_for_data_point(self, prediction_data_point: TextPair) -> torch.Tensor:
        embedding_names = self.embeddings.get_names()
        if self.embed_separately:
            self.embeddings.embed([prediction_data_point.first, prediction_data_point.second])
            return torch.cat(
                [
                    prediction_data_point.first.get_embedding(embedding_names),
                    prediction_data_point.second.get_embedding(embedding_names),
                ],
                0,
            )
        else:
            concatenated_sentence = Sentence(
                prediction_data_point.first.to_tokenized_string()
                + self.sep
                + prediction_data_point.second.to_tokenized_string(),
                use_tokenizer=False,
            )
            self.embeddings.embed(concatenated_sentence)
            return concatenated_sentence.get_embedding(embedding_names)

    def _encode_data_points(self, sentence_pairs: List[TextPair], data_points: List[DT2]):

        embedding_names = self.embeddings.get_names()

        if self.embed_separately:  # embed both sentences separately, concatenate the resulting vectors
            first_elements = [pair.first for pair in sentence_pairs]
            second_elements = [pair.second for pair in sentence_pairs]

            self.embeddings.embed(first_elements)
            self.embeddings.embed(second_elements)

            text_embedding_list = [
                torch.cat(
                    [
                        a.get_embedding(embedding_names),
                        b.get_embedding(embedding_names),
                    ],
                    0,
                ).unsqueeze(0)
                for (a, b) in zip(first_elements, second_elements)
            ]

        else:  # concatenate two sentences and embed together
            concatenated_sentences = [
                Sentence(
                    pair.first.to_tokenized_string() + self.sep + pair.second.to_tokenized_string(),
                    use_tokenizer=False,
                )
                for pair in sentence_pairs
            ]

            self.embeddings.embed(concatenated_sentences)

            text_embedding_list = [
                sentence.get_embedding(embedding_names).unsqueeze(0) for sentence in concatenated_sentences
            ]

        # get a tensor of data points
        data_point_tensor = torch.stack(text_embedding_list)

        # do dropout
        data_point_tensor = self.dropout(data_point_tensor)
        data_point_tensor = self.locked_dropout(data_point_tensor)
        data_point_tensor = self.word_dropout(data_point_tensor)
        data_point_tensor = data_point_tensor.squeeze(1)

        return data_point_tensor

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "embeddings": self.embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "multi_label": self.multi_label,
            "multi_label_threshold": self.multi_label_threshold,
            "weight_dict": self.weight_dict,
            "embed_separately": self.embed_separately,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            multi_label=state.get("multi_label_threshold", 0.5),
            loss_weights=state.get("weight_dict"),
            embed_separately=state.get("embed_separately"),
            **kwargs,
        )
