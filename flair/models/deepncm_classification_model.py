import logging
from typing import Literal, Optional

import torch

import flair
from flair.data import Dictionary

log = logging.getLogger("flair")


class DeepNCMDecoder(torch.nn.Module):
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
        label_dictionary: Dictionary,
        embeddings_size: int,
        encoding_dim: Optional[int] = None,
        alpha: float = 0.9,
        mean_update_method: Literal["online", "condensation", "decay"] = "online",
        use_encoder: bool = True,
        multi_label: bool = False,  # should get from the Model it belongs to
    ) -> None:
        """Initialize a DeepNCMDecoder.

        Args:
            encoding_dim: The dimensionality of the encoded embeddings (default is the same as the input embeddings).
            alpha: The decay factor for updating class prototypes (default is 0.9). This only applies when mean_update_method is 'decay'.
            mean_update_method: The method for updating class prototypes ('online', 'condensation', or 'decay').
            use_encoder: Whether to apply an encoder to the input embeddings (default is True).
            multi_label: Whether to predict multiple labels per sentence (default is False).
        """

        super().__init__()

        self.label_dictionary = label_dictionary
        self._num_prototypes = len(label_dictionary)

        self.alpha = alpha
        self.mean_update_method = mean_update_method
        self.use_encoder = use_encoder
        self.multi_label = multi_label

        self.embedding_dim = embeddings_size

        if use_encoder:
            self.encoding_dim = encoding_dim or self.embedding_dim
        else:
            self.encoding_dim = self.embedding_dim

        self.class_prototypes = torch.nn.Parameter(
            torch.nn.functional.normalize(torch.randn(self._num_prototypes, self.encoding_dim)), requires_grad=False
        )

        self.class_counts = torch.nn.Parameter(torch.zeros(self._num_prototypes), requires_grad=False)
        self.prototype_updates = torch.zeros_like(self.class_prototypes).to(flair.device)
        self.prototype_update_counts = torch.zeros(self._num_prototypes).to(flair.device)
        self.to(flair.device)

        self._validate_parameters()

        if self.use_encoder:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, self.encoding_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.encoding_dim * 2, self.encoding_dim),
            )
        else:
            self.encoder = torch.nn.Sequential(torch.nn.Identity())

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    def _validate_parameters(self) -> None:
        """Validate that the input parameters have valid and compatible values."""
        assert 0 <= self.alpha <= 1, "alpha must be in the range [0, 1]"
        assert self.mean_update_method in [
            "online",
            "condensation",
            "decay",
        ], f"Invalid mean_update_method: {self.mean_update_method}. Must be 'online', 'condensation', or 'decay'"
        assert self.encoding_dim > 0, "encoding_dim must be greater than 0"

    @property
    def num_prototypes(self) -> int:
        """The number of class prototypes."""
        return self.class_prototypes.size(0)

    def _calculate_distances(self, encoded_embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate the squared Euclidean distance between encoded embeddings and class prototypes.

        Args:
            encoded_embeddings: Encoded representations of the input sentences.

        Returns:
            torch.Tensor: Distances between encoded embeddings and class prototypes.
        """
        return torch.cdist(encoded_embeddings, self.class_prototypes).pow(2)

    def _calculate_prototype_updates(self, encoded_embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Calculate updates for class prototypes based on the current batch.

        Args:
            encoded_embeddings: Encoded representations of the input sentences.
            labels: True labels for the input sentences.
        """
        one_hot = (
            labels if self.multi_label else torch.nn.functional.one_hot(labels, num_classes=self.num_prototypes).float()
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
            self.prototype_update_counts = torch.zeros(self.num_prototypes, device=flair.device)

    def forward(self, embedded: torch.Tensor, label_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the decoder, which calculates the scores as prototype distances.

        :param embedded: Embedded representations of the input sentences.
        :param label_tensor: True labels for the input sentences as a tensor.
        :return: Scores as a tensor of distances to class prototypes.
        """
        encoded_embeddings = self.encoder(embedded)

        distances = self._calculate_distances(encoded_embeddings)

        if label_tensor is not None:
            self._calculate_prototype_updates(encoded_embeddings, label_tensor)

        scores = -distances

        return scores

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

    def get_closest_prototypes(self, input_vector: torch.Tensor, top_k: int = 5) -> list[tuple[str, float]]:
        """Get the k closest prototype vectors to the given input vector using the configured distance metric.

        Args:
            input_vector (torch.Tensor): The input vector to compare against prototypes.
            top_k (int): The number of closest prototypes to return (default is 5).

        Returns:
            list[tuple[str, float]]: Each tuple contains (class_name, distance).
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
