import typing

import torch

import flair.embeddings
import flair.nn
from flair.data import Corpus, Sentence, TextPair, _iter_dataset


class TextPairClassifier(flair.nn.DefaultClassifier[TextPair, TextPair]):
    """Text Pair Classification Model for tasks such as Recognizing Textual Entailment, build upon TextClassifier.

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
    ) -> None:
        """Initializes a TextPairClassifier.

        Args:
            label_type: label_type: name of the label
            embed_separately: if True, the sentence embeddings will be concatenated,
              if False both sentences will be combined and newly embedded.
            embeddings: embeddings used to embed each data point
            label_dictionary: dictionary of labels you want to predict
            multi_label: auto-detected by default, but you can set this to True to force multi-label prediction
               or False to force single-label prediction
            multi_label_threshold: If multi-label you can set the threshold to make predictions
            loss_weights: Dictionary of weights for labels for the loss function.
              If any label's weight is unspecified it will default to 1.0
            **classifierargs: The arguments propagated to :meth:`flair.nn.DefaultClassifier.__init__`
        """
        super().__init__(
            **classifierargs,
            embeddings=embeddings,
            final_embedding_size=2 * embeddings.embedding_length if embed_separately else embeddings.embedding_length,
            should_embed_sentence=False,
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

    def _get_data_points_from_sentence(self, sentence: TextPair) -> list[TextPair]:
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
            # If the concatenated version of the text pair does not exist yet, create it
            if prediction_data_point.concatenated_data is None:
                concatenated_sentence = Sentence(
                    prediction_data_point.first.to_tokenized_string()
                    + self.sep
                    + prediction_data_point.second.to_tokenized_string(),
                    use_tokenizer=False,
                )
                prediction_data_point.concatenated_data = concatenated_sentence
            self.embeddings.embed(prediction_data_point.concatenated_data)
            return prediction_data_point.concatenated_data.get_embedding(embedding_names)

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "document_embeddings": self.embeddings.save_embeddings(use_state_dict=False),
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "embed_separately": self.embed_separately,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("document_embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            embed_separately=state.get("embed_separately"),
            **kwargs,
        )

    def get_used_tokens(
        self, corpus: Corpus, context_length: int = 0, respect_document_boundaries: bool = True
    ) -> typing.Iterable[list[str]]:
        for sentence_pair in _iter_dataset(corpus.get_all_sentences()):
            yield [t.text for t in sentence_pair.first]
            yield [t.text for t in sentence_pair.first.left_context(context_length, respect_document_boundaries)]
            yield [t.text for t in sentence_pair.first.right_context(context_length, respect_document_boundaries)]
            yield [t.text for t in sentence_pair.second]
            yield [t.text for t in sentence_pair.second.left_context(context_length, respect_document_boundaries)]
            yield [t.text for t in sentence_pair.second.right_context(context_length, respect_document_boundaries)]
