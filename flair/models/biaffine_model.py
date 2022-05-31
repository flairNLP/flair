import logging

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn
from tqdm import tqdm

import flair.nn
from flair.data import Sentence, Dictionary, Span
from flair.embeddings import TokenEmbeddings
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.training_utils import store_embeddings

from .sequence_tagger_utils.biaffine import Biaffine, BiaffineDecoder

log = logging.getLogger("flair")

class BiaffineTager(flair.nn.Classifier[Sentence]):
    def __init__(
            self,
            embeddings: TokenEmbeddings,
            tag_dictionary: Dictionary,
            tag_type: str,
            ffnn_output_size: int = 150,
            use_biaffine: bool = True,
            dropout: float = 0.0,
            ffnn_dropout: float = 0.2,
            is_flat_ner: bool = False,
            loss_weights: Dict[str, float] = None,
            init_from_state_dict: bool = False,
    ):

        super(BiaffineTager, self).__init__()

        self.tag_type = tag_type

        if init_from_state_dict:
            self.label_dictionary = tag_dictionary
        else:
            tag_dictionary.add_item('O')
            self.label_dictionary = tag_dictionary
        self.tagset_size = len(self.label_dictionary)
        log.info(f"SequenceTagger predicts: {self.label_dictionary}")

        self.embeddings = embeddings
        embedding_dim: int = embeddings.embedding_length

        # ----- Initial loss weights parameters -----
        self.weight_dict = loss_weights
        self.loss_weights = self._init_loss_weights(loss_weights) if loss_weights else None

        self.use_biaffine = use_biaffine
        self.ffnn_size = ffnn_output_size
        self.ffnn_dropout = ffnn_dropout
        self.is_flat_ner = is_flat_ner

        self.use_dropout: float = dropout

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        ffnn_input_size = embedding_dim

        if use_biaffine:
            self.biaffine = Biaffine(ffnn_input_size, ffnn_output_size, ffnn_dropout, len(tag_dictionary), init_from_state_dict)
            self.biaffine_decoder = BiaffineDecoder(self.label_dictionary)
            self.loss_function = torch.nn.CrossEntropyLoss(weight=self.loss_weights, reduction="sum")

        self.to(flair.device)

    @property
    def label_type(self):
        return self.tag_type

    def _init_loss_weights(self, loss_weights: Dict[str, float]) -> torch.Tensor:
        """
        Intializes the loss weights based on given dictionary:
        :param loss_weights: dictionary - contains loss weights
        """
        n_classes = len(self.label_dictionary)
        weight_list = [1.0 for _ in range(n_classes)]
        for i, tag in enumerate(self.label_dictionary.get_items()):
            if tag in loss_weights.keys():
                weight_list[i] = loss_weights[tag]

        return torch.tensor(weight_list).to(flair.device)

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> Tuple[torch.Tensor, int]:

        # if there are no sentences, there is no loss
        if len(sentences) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0

        # forward pass to get scores
        scores, gold_labels = self.forward(sentences)  # type: ignore

        # calculate loss given scores and labels
        return self._calculate_loss(scores, gold_labels)

    def forward(self, sentences: Union[List[Sentence], Sentence]):

        self.embeddings.embed(sentences)

        lengths, sentence_tensor = self._make_padded_tensor_for_batch(sentences)

        lengths = lengths.sort(dim=0, descending=True)
        sentences = [sentences[i] for i in lengths.indices]
        sentence_tensor = sentence_tensor[lengths.indices]

        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)

        candidate = self.biaffine(sentence_tensor)
        gold_labels = self.biaffine_decoder.get_labels(sentences)
        scores = self.biaffine_decoder.get_flat_scores(lengths, candidate)

        return scores, gold_labels

    def _calculate_loss(self, scores, labels) -> Tuple[torch.Tensor, int]:

        if not any(labels):
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        labels = torch.tensor(
            [
                self.label_dictionary.get_idx_for_item(label[0])
                if len(label) > 0
                else self.label_dictionary.get_idx_for_item("O")
                for label in labels
            ],
            dtype=torch.long,
            device=flair.device,
        )

        return self.loss_function(scores, labels), len(labels)


    def _make_padded_tensor_for_batch(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, torch.Tensor]:
        names = self.embeddings.get_names()
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
            )
        all_embs = list()
        for sentence in sentences:
            all_embs += [emb for token in sentence for emb in token.get_each_embedding(names)]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[: self.embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)
        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )
        lengths: torch.Tensor = torch.tensor(lengths, dtype=torch.long)
        return lengths, sentence_tensor

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            mini_batch_size: int = 32,
            return_probabilities_for_all_classes: bool = False,
            verbose: bool = False,
            label_name: Optional[str] = None,
            return_loss=False,
            embedding_storage_mode="none",
    ):

        if label_name is None:
            label_name = self.tag_type

        with torch.no_grad():
            if not sentences:
                return sentences

            # make sure its a list
            if not isinstance(sentences, list) and not isinstance(sentences, flair.data.Dataset):
                sentences = [sentences]

            # filter empty sentences
            sentences = [sentence for sentence in sentences if len(sentence) > 0]

            # reverse sort all sequences by their length
            reordered_sentences = sorted(sentences, key=lambda s: len(s), reverse=True)

            if len(reordered_sentences) == 0:
                return sentences

            dataloader = DataLoader(
                dataset=FlairDatapointDataset(reordered_sentences),
                batch_size=mini_batch_size,
            )
            if verbose:
                dataloader = tqdm(dataloader)

            overall_loss = 0
            batch_no = 0
            label_count = 0

            for batch in dataloader:

                batch_no += 1

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {batch_no}")

                # stop if all sentences are empty
                if not batch:
                    continue

                # get features from forward propagation
                features, gold_labels = self.forward(batch)

                for sentence in batch:
                    sentence.remove_labels(label_name)

                if return_loss:
                    loss = self._calculate_loss(features, gold_labels)
                    overall_loss += loss[0]
                    label_count += loss[1]

                # Sort batch in same way as forward propagation
                lengths = torch.LongTensor([len(sentence) for sentence in batch])
                lengths = lengths.sort(dim=0, descending=True)
                batch = [batch[i] for i in lengths.indices]

                predictions = self.biaffine_decoder.decode(features, batch, self.is_flat_ner)

                for sentence, sentence_predictions in zip(batch, predictions):
                    for predicted_span in sentence_predictions:
                        span: Span = sentence[predicted_span[0][0] : predicted_span[0][-1] + 1]
                        span.add_label(label_name, value=predicted_span[2], score=predicted_span[1])

                store_embeddings(sentences, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count

    def _get_state_dict(self):
        """Returns the state dictionary for this model."""
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "ffnn_size": self.ffnn_size,
            "tag_dictionary": self.label_dictionary,
            "tag_type": self.tag_type,
            "use_biaffine": self.use_biaffine,
            "use_dropout": self.use_dropout,
            "ffnn_dropout": self.ffnn_dropout,
            "is_flat_ner": self.is_flat_ner,
            "weight_dict": self.weight_dict,
        }

        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        """Initialize the model from a state dictionary."""
        use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]

        model = BiaffineTager(
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            use_biaffine=state["use_biaffine"],
            ffnn_output_size=state["ffnn_size"],
            dropout=use_dropout,
            ffnn_dropout=state["ffnn_dropout"],
            is_flat_ner=state["is_flat_ner"],
            loss_weights=weights,
            init_from_state_dict=True,
        )

        model.load_state_dict(state["state_dict"])
        return model