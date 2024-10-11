import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from tqdm import tqdm

import flair
from flair.data import Dictionary, Sentence
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import DocumentEmbeddings
from flair.embeddings.base import load_embeddings
from flair.nn import Classifier

log = logging.getLogger("flair")


class DeepNCMClassifier(Classifier[Sentence]):
    """Deep Nearest Class Mean (DeepNCM) Classifier for text classification tasks.

    This model combines deep learning with the Nearest Class Mean (NCM) approach.
    It uses document embeddings to represent text, optionally applies an encoder,
    and classifies based on the nearest class prototype in the embedded space.

    The model supports various methods for updating class prototypes during training,
    making it adaptable to different learning scenarios.

    This implementation is based on the research paper:
    Guerriero, S., Caputo, B., & Mensink, T. (2018). DeepNCM: Deep Nearest Class Mean Classifiers.
    In International Conference on Learning Representations (ICLR) 2018 Workshop.
    URL: https://openreview.net/forum?id=rkPLZ4JPM
    """

    def __init__(
        self,
        embeddings: DocumentEmbeddings,
        label_dictionary: Dictionary,
        label_type: str,
        encoding_dim: Optional[int] = None,
        alpha: float = 0.9,
        mean_update_method: Literal["online", "condensation", "decay"] = "online",
        use_encoder: bool = True,
        multi_label: bool = False,
        multi_label_threshold: float = 0.5,
    ):
        """Initialize a DeepNCMClassifier.

        Args:
            embeddings: Document embeddings to use for encoding text.
            label_dictionary: Dictionary containing the label vocabulary.
            label_type: The type of label to predict.
            encoding_dim: The dimensionality of the encoded embeddings (default is the same as the input embeddings).
            alpha: The decay factor for updating class prototypes (default is 0.9).
            mean_update_method: The method for updating class prototypes ('online', 'condensation', or 'decay').
            use_encoder: Whether to apply an encoder to the input embeddings (default is True).
            multi_label: Whether to predict multiple labels per sentence (default is False).
            multi_label_threshold: The threshold for multi-label prediction (default is 0.5).
        """
        super().__init__()

        self.embeddings = embeddings
        self.label_dictionary = label_dictionary
        self._label_type = label_type
        self.alpha = alpha
        self.mean_update_method = mean_update_method
        self.use_encoder = use_encoder
        self.multi_label = multi_label
        self.multi_label_threshold = multi_label_threshold
        self.num_classes = len(label_dictionary)
        self.embedding_dim = embeddings.embedding_length

        if use_encoder:
            self.encoding_dim = encoding_dim or self.embedding_dim
        else:
            self.encoding_dim = self.embedding_dim

        self._validate_parameters()

        if self.use_encoder:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, self.encoding_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.encoding_dim * 2, self.encoding_dim),
            )
        else:
            self.encoder = torch.nn.Sequential(torch.nn.Identity())

        self.loss_function = (
            torch.nn.BCEWithLogitsLoss(reduction="sum")
            if self.multi_label
            else torch.nn.CrossEntropyLoss(reduction="sum")
        )

        self.class_prototypes = torch.nn.Parameter(
            torch.nn.functional.normalize(torch.randn(self.num_classes, self.encoding_dim)), requires_grad=False
        )
        self.class_counts = torch.nn.Parameter(torch.zeros(self.num_classes), requires_grad=False)
        self.prototype_updates = torch.zeros_like(self.class_prototypes).to(flair.device)
        self.prototype_update_counts = torch.zeros(self.num_classes).to(flair.device)
        self.to(flair.device)

    def _validate_parameters(self) -> None:
        """Validate the input parameters."""
        assert 0 <= self.alpha <= 1, "alpha must be in the range [0, 1]"
        assert self.mean_update_method in [
            "online",
            "condensation",
            "decay",
        ], f"Invalid mean_update_method: {self.mean_update_method}. Must be 'online', 'condensation', or 'decay'"
        assert self.encoding_dim > 0, "encoding_dim must be greater than 0"

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """Encode the input sentences using embeddings and optional encoder.

        Args:
            sentences: Input sentence or list of sentences.

        Returns:
            torch.Tensor: Encoded representations of the input sentences.
        """
        if not isinstance(sentences, list):
            sentences = [sentences]

        self.embeddings.embed(sentences)
        sentence_embeddings = torch.stack([sentence.get_embedding() for sentence in sentences])
        encoded_embeddings = self.encoder(sentence_embeddings)

        return encoded_embeddings

    def _calculate_distances(self, encoded_embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate distances between encoded embeddings and class prototypes.

        Args:
            encoded_embeddings: Encoded representations of the input sentences.

        Returns:
            torch.Tensor: Distances between encoded embeddings and class prototypes.
        """
        return torch.cdist(encoded_embeddings, self.class_prototypes)

    def forward_loss(self, data_points: List[Sentence]) -> Tuple[torch.Tensor, int]:
        """Compute the loss for a batch of sentences.

        Args:
            data_points: A list of sentences.

        Returns:
            Tuple[torch.Tensor, int]: The total loss and the number of sentences.
        """
        encoded_embeddings = self.forward(data_points)
        labels = self._prepare_label_tensor(data_points)
        distances = self._calculate_distances(encoded_embeddings)
        loss = self.loss_function(-distances, labels)
        self._calculate_prototype_updates(encoded_embeddings, labels)

        return loss, len(data_points)

    def _prepare_label_tensor(self, sentences: List[Sentence]) -> torch.Tensor:
        """Prepare the label tensor for the given sentences.

        Args:
            sentences: A list of sentences.

        Returns:
            torch.Tensor: The label tensor for the given sentences.
        """
        if self.multi_label:
            return torch.tensor(
                [
                    [
                        (
                            1
                            if label
                            in [sentence_label.value for sentence_label in sentence.get_labels(self._label_type)]
                            else 0
                        )
                        for label in self.label_dictionary.get_items()
                    ]
                    for sentence in sentences
                ],
                dtype=torch.float,
                device=flair.device,
            )
        else:
            return torch.tensor(
                [
                    self.label_dictionary.get_idx_for_item(sentence.get_label(self._label_type).value)
                    for sentence in sentences
                ],
                dtype=torch.long,
                device=flair.device,
            )

    def _calculate_prototype_updates(self, encoded_embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Calculate updates for class prototypes based on the current batch.

        Args:
            encoded_embeddings: Encoded representations of the input sentences.
            labels: True labels for the input sentences.
        """
        one_hot = (
            labels if self.multi_label else torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        )

        updates = torch.matmul(one_hot.t(), encoded_embeddings)
        counts = one_hot.sum(dim=0)
        mask = counts > 0
        self.prototype_updates[mask] += updates[mask]
        self.prototype_update_counts[mask] += counts[mask]

    def update_prototypes(self) -> None:
        """Apply accumulated updates to class prototypes."""
        with torch.no_grad():
            update_mask = self.prototype_update_counts > 0
            if update_mask.any():
                if self.mean_update_method in ["online", "condensation"]:
                    new_counts = self.class_counts[update_mask] + self.prototype_update_counts[update_mask]
                    self.class_prototypes[update_mask] = (
                        self.class_counts[update_mask].unsqueeze(1) * self.class_prototypes[update_mask]
                        + self.prototype_updates[update_mask]
                    ) / new_counts.unsqueeze(1)
                    self.class_counts[update_mask] = new_counts
                elif self.mean_update_method == "decay":
                    new_prototypes = self.prototype_updates[update_mask] / self.prototype_update_counts[
                        update_mask
                    ].unsqueeze(1)
                    self.class_prototypes[update_mask] = (
                        self.alpha * self.class_prototypes[update_mask] + (1 - self.alpha) * new_prototypes
                    )
                    self.class_counts[update_mask] += self.prototype_update_counts[update_mask]

            # Reset prototype updates
            self.prototype_updates = torch.zeros_like(self.class_prototypes, device=flair.device)
            self.prototype_update_counts = torch.zeros(self.num_classes, device=flair.device)

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss: bool = False,
        embedding_storage_mode: str = "none",
    ) -> Union[List[Sentence], Tuple[float, int]]:
        """Predict classes for a list of sentences.

        Args:
            sentences: A list of sentences or a single sentence.
            mini_batch_size: Size of mini batches during prediction.
            return_probabilities_for_all_classes: Whether to return probabilities for all classes.
            verbose: If True, show progress bar during prediction.
            label_name: The name of the label to use for prediction.
            return_loss: If True, compute and return loss.
            embedding_storage_mode: The mode for storing embeddings ('none', 'cpu', or 'gpu').

        Returns:
            Union[List[Sentence], Tuple[float, int]]:
                if return_loss is True, returns a tuple of total loss and total number of sentences;
                otherwise, returns the list of sentences with predicted labels.
        """
        with torch.no_grad():
            if not isinstance(sentences, list):
                sentences = [sentences]
            if not sentences:
                return sentences

            label_name = label_name or self.label_type
            Sentence.set_context_for_sentences(sentences)

            filtered_sentences = [sent for sent in sentences if len(sent) > 0]
            reordered_sentences = sorted(filtered_sentences, key=len, reverse=True)

            if len(reordered_sentences) == 0:
                return sentences

            dataloader = DataLoader(
                dataset=FlairDatapointDataset(reordered_sentences),
                batch_size=mini_batch_size,
            )

            if verbose:
                progress_bar = tqdm(dataloader)
                progress_bar.set_description("Predicting")
                dataloader = progress_bar

            total_loss = 0.0
            total_sentences = 0

            for batch in dataloader:
                if not batch:
                    continue

                encoded_embeddings = self.forward(batch)
                distances = self._calculate_distances(encoded_embeddings)

                if self.multi_label:
                    probabilities = torch.sigmoid(-distances)
                else:
                    probabilities = torch.nn.functional.softmax(-distances, dim=1)

                if return_loss:
                    labels = self._prepare_label_tensor(batch)
                    loss = self.loss_function(-distances, labels)
                    total_loss += loss.item()
                    total_sentences += len(batch)

                for sentence_index, sentence in enumerate(batch):
                    sentence.remove_labels(label_name)

                    if self.multi_label:
                        for label_index, probability in enumerate(probabilities[sentence_index]):
                            if probability > self.multi_label_threshold or return_probabilities_for_all_classes:
                                label_value = self.label_dictionary.get_item_for_index(label_index)
                                sentence.add_label(label_name, label_value, probability.item())
                    else:
                        predicted_idx = torch.argmax(probabilities[sentence_index])
                        label_value = self.label_dictionary.get_item_for_index(predicted_idx.item())
                        sentence.add_label(label_name, label_value, probabilities[sentence_index, predicted_idx].item())

                        if return_probabilities_for_all_classes:
                            for label_index, probability in enumerate(probabilities[sentence_index]):
                                label_value = self.label_dictionary.get_item_for_index(label_index)
                                sentence.add_label(f"{label_name}_all", label_value, probability.item())

                for sentence in batch:
                    sentence.clear_embeddings(embedding_storage_mode)

        if return_loss:
            return total_loss, total_sentences
        return sentences

    def _get_state_dict(self) -> Dict[str, Any]:
        """Get the state dictionary of the model.

        Returns:
            Dict[str, Any]: The state dictionary containing model parameters and configuration.
        """
        model_state = {
            "embeddings": self.embeddings.save_embeddings(),
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "encoding_dim": self.encoding_dim,
            "alpha": self.alpha,
            "mean_update_method": self.mean_update_method,
            "use_encoder": self.use_encoder,
            "multi_label": self.multi_label,
            "multi_label_threshold": self.multi_label_threshold,
            "class_prototypes": self.class_prototypes.cpu(),
            "class_counts": self.class_counts.cpu(),
            "encoder": self.encoder.state_dict(),
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs) -> "DeepNCMClassifier":
        """Initialize the model from a state dictionary.

        Args:
            state: The state dictionary containing model parameters and configuration.
            **kwargs: Additional keyword arguments for model initialization.

        Returns:
            DeepNCMClassifier: An instance of the model initialized with the given state.
        """
        embeddings = state["embeddings"]
        if isinstance(embeddings, dict):
            embeddings = load_embeddings(embeddings)

        model = cls(
            embeddings=embeddings,
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            encoding_dim=state["encoding_dim"],
            alpha=state["alpha"],
            mean_update_method=state["mean_update_method"],
            use_encoder=state["use_encoder"],
            multi_label=state.get("multi_label", False),
            multi_label_threshold=state.get("multi_label_threshold", 0.5),
            **kwargs,
        )

        if "encoder" in state:
            model.encoder.load_state_dict(state["encoder"])
        if "class_prototypes" in state:
            model.class_prototypes.data = state["class_prototypes"].to(flair.device)
        if "class_counts" in state:
            model.class_counts.data = state["class_counts"].to(flair.device)

        return model

    def get_prototype(self, class_name: str) -> torch.Tensor:
        """Get the prototype vector for a given class name.

        Args:
            class_name: The name of the class whose prototype vector is requested.

        Returns:
            torch.Tensor: The prototype vector for the given class.

        Raises:
            ValueError: If the class name is not found in the label dictionary.
        """
        try:
            class_idx = self.label_dictionary.get_idx_for_item(class_name)
        except IndexError as exc:
            raise ValueError(f"Class name '{class_name}' not found in the label dictionary") from exc

        return self.class_prototypes[class_idx].clone()

    def get_closest_prototypes(self, input_vector: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get the top_k closest prototype vectors to the given input vector using the configured distance metric.

        Args:
            input_vector (torch.Tensor): The input vector to compare against prototypes.
            top_k (int): The number of closest prototypes to return (default is 5).

        Returns:
            List[Tuple[str, float]]: Each tuple contains (class_name, distance).
        """
        if input_vector.dim() != 1:
            raise ValueError("Input vector must be a 1D tensor")
        if input_vector.size(0) != self.class_prototypes.size(1):
            raise ValueError(
                f"Input vector dimension ({input_vector.size(0)}) does not match prototype dimension ({self.class_prototypes.size(1)})"
            )

        input_vector = input_vector.unsqueeze(0)
        distances = self._calculate_distances(input_vector)
        top_k_values, top_k_indices = torch.topk(distances.squeeze(), k=top_k, largest=False)

        nearest_prototypes = []
        for idx, value in zip(top_k_indices, top_k_values):
            class_name = self.label_dictionary.get_item_for_index(idx.item())
            nearest_prototypes.append((class_name, value.item()))

        return nearest_prototypes

    @property
    def label_type(self) -> str:
        """Get the label type for this classifier."""
        return self._label_type

    def __str__(self) -> str:
        """Get a string representation of the model.

        Returns:
            str: A string describing the model architecture.
        """
        return (
            f"DeepNCMClassifier(\n"
            f"  (embeddings): {self.embeddings}\n"
            f"  (encoder): {self.encoder}\n"
            f"  (prototypes): {self.class_prototypes.shape}\n"
            f")"
        )
