from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import flair
from flair.data import Dictionary, Sentence, _iter_dataset
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import DocumentEmbeddings
from flair.training_utils import store_embeddings


class OneClassClassifier(flair.nn.Classifier[Sentence]):
    """One Class Classification Model for tasks such as Anomaly Detection.

    Task
    ----
    One Class Classification (OCC) tries to identify objects of a specific class amongst all objects, in contrast to
    distinguishing between two or more classes.

    Example:
    -------
    The model expects to be trained on a dataset in which every element has the same label_value, e.g. movie reviews
    with the label POSITIVE.
    During inference, one of two label_values will be added:
    - In-class (e.g. another movie review) -> label_value="POSITIVE"
    - Anything else (e.g. a wiki page)     -> label_value="<unk>"

    Architecture
    ------------
    Reconstruction with autoencoder. The score is the reconstruction error from compressing and decompressing the
    document embedding. A LOWER score indicates a HIGHER probability of being in-class. The threshold is
    calculated as a high percentile of the score distribution of in-class elements from the dev set.

    You must set the threshold after training by running `model.threshold = model.calculate_threshold(corpus.dev)`.
    """

    def __init__(
        self,
        embeddings: DocumentEmbeddings,
        label_dictionary: Dictionary,
        label_type: str,
        encoding_dim: int = 128,
        threshold: Optional[float] = None,
    ) -> None:
        """Initializes a OneClassClassifier.

        Args:
        embeddings: Embeddings to use during training and prediction
        label_dictionary: The label to predict. Must contain exactly one class.
        label_type: name of the annotation_layer to be predicted in case a corpus has multiple annotations
        encoding_dim: The size of the compressed embedding
        threshold: The score that separates in-class from out-of-class
        """
        super().__init__()
        self.embeddings = embeddings
        if len(label_dictionary) != 1:
            raise ValueError(f"label_dictionary must have exactly 1 element: {label_dictionary}")
        self.label_dictionary = label_dictionary
        self.label_value = label_dictionary.get_items()[0]
        self._label_type = label_type
        self.encoding_dim = encoding_dim
        self.threshold = threshold

        embedding_dim = embeddings.embedding_length
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, encoding_dim * 4),
            torch.nn.LeakyReLU(True),
            torch.nn.Linear(encoding_dim * 4, encoding_dim * 2),
            torch.nn.LeakyReLU(True),
            torch.nn.Linear(encoding_dim * 2, encoding_dim),
            torch.nn.LeakyReLU(True),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, encoding_dim * 2),
            torch.nn.LeakyReLU(True),
            torch.nn.Linear(encoding_dim * 2, encoding_dim * 4),
            torch.nn.LeakyReLU(True),
            torch.nn.Linear(encoding_dim * 4, embedding_dim),
            torch.nn.LeakyReLU(True),
        )

        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)
        self.to(flair.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_loss(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, int]:
        """Returns Tuple[scalar tensor, num examples]."""
        if len(sentences) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0
        sentence_tensor = self._sentences_to_tensor(sentences)
        reconstructed_sentence_tensor = self.forward(sentence_tensor)
        return self._calculate_loss(reconstructed_sentence_tensor, sentence_tensor).sum(), len(sentences)

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ) -> Optional[torch.Tensor]:
        """Predicts the class labels for the given sentences. The labels are directly added to the sentences.

        Args:
            sentences: list of sentences to predict
            mini_batch_size: the amount of sentences that will be predicted within one batch
            return_probabilities_for_all_classes: doesn't do anything for this class
            verbose: set to True to display a progress bar
            return_loss: set to True to return loss
            label_name: set this to change the name of the label type that is predicted
            embedding_storage_mode: default is 'none' which is the best is most cases.
                Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively. 'gpu' to store embeddings in GPU memory.

        Returns: None. If return_loss is set, returns a scalar tensor
        """
        if label_name is None:
            label_name = self.label_type

        with torch.no_grad():
            # make sure it's a list
            if not isinstance(sentences, list):
                sentences = [sentences]

            Sentence.set_context_for_sentences(cast(List[Sentence], sentences))

            # filter empty sentences
            sentences = [sentence for sentence in sentences if len(sentence) > 0]
            if len(sentences) == 0:
                return torch.tensor(0.0, requires_grad=True, device=flair.device) if return_loss else None

            dataloader = DataLoader(
                dataset=FlairDatapointDataset(sentences),
                batch_size=mini_batch_size,
            )
            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader, desc="Batch inference")

            overall_loss = torch.zeros(1, device=flair.device)
            for batch in dataloader:
                # stop if all sentences are empty
                if not batch:
                    continue

                sentence_tensor = self._sentences_to_tensor(batch)
                reconstructed = self.forward(sentence_tensor)
                loss_tensor = self._calculate_loss(reconstructed, sentence_tensor)

                for sentence, loss in zip(batch, loss_tensor.tolist()):
                    sentence.remove_labels(label_name)
                    label_value = self.label_value if self.threshold is not None and loss < self.threshold else "<unk>"
                    sentence.add_label(typename=label_name, value=label_value, score=loss)

                overall_loss += loss_tensor.sum()
                store_embeddings(batch, storage_mode=embedding_storage_mode)

        return overall_loss if return_loss else None

    @property
    def label_type(self) -> str:
        return self._label_type

    def _sentences_to_tensor(self, sentences: List[Sentence]) -> torch.Tensor:
        self.embeddings.embed(sentences)
        return torch.stack([sentence.embedding for sentence in sentences])

    def _calculate_loss(self, predicted: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return cosine similarity loss.

        Args:
            predicted: tensor of shape (batch_size, embedding_size)
            labels: tensor of shape (batch_size, embedding_size)

        Returns:
            tensor of shape (batch_size)
        """
        if labels.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device)

        return 1 - self.cosine_sim(predicted, labels)

    def _get_state_dict(self):
        """Returns the state dictionary for this model."""
        model_state = {
            **super()._get_state_dict(),
            "embeddings": self.embeddings.save_embeddings(use_state_dict=False),
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "encoding_dim": self.encoding_dim,
            "threshold": self.threshold,
        }

        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            encoding_dim=state.get("encoding_dim"),
            threshold=state.get("threshold"),
            **kwargs,
        )

    @classmethod
    def load(cls, model_path: Union[str, Path, Dict[str, Any]]) -> "OneClassClassifier":
        from typing import cast

        return cast("OneClassClassifier", super().load(model_path=model_path))

    def calculate_threshold(self, dataset: Dataset[Sentence], quantile=0.995) -> float:
        """Determine the score threshold to consider a Sentence in-class.

        This implementation returns the score at which `quantile` of `dataset` will be considered in-class. Intended
        for use-cases desiring high-recall.
        """

        def score(sentence: Sentence) -> float:
            sentence_tensor = self._sentences_to_tensor([sentence])
            reconstructed = self.forward(sentence_tensor)
            loss_tensor = self._calculate_loss(reconstructed, sentence_tensor)
            return loss_tensor.tolist()[0]

        scores = [
            score(sentence)
            for sentence in _iter_dataset(dataset)
            if sentence.get_labels(self.label_type)[0].value == self.label_value
        ]
        threshold = np.quantile(scores, quantile)
        return threshold
