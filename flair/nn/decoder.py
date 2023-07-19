import logging
from typing import List, Optional, Dict

import torch
import random
import numpy as np
from pathlib import Path

import flair
from flair.data import Dictionary, Sentence
from flair.embeddings import Embeddings
from flair.nn.distance import (
    CosineDistance,
    EuclideanDistance,
    HyperbolicDistance,
    LogitCosineDistance,
    NegativeScaledDotProduct,
)
from flair.training_utils import store_embeddings

import wikidata_NER_mapping

logger = logging.getLogger("flair")


class PrototypicalDecoder(torch.nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        embeddings_size: int,
        prototype_size: Optional[int] = None,
        distance_function: str = "euclidean",
        use_radius: Optional[bool] = False,
        min_radius: Optional[int] = 0,
        unlabeled_distance: Optional[float] = None,
        unlabeled_idx: Optional[int] = None,
        learning_mode: Optional[str] = "joint",
        normal_distributed_initial_prototypes: bool = False,
    ) -> None:
        super().__init__()

        if not prototype_size:
            prototype_size = embeddings_size

        self.prototype_size = prototype_size

        # optional metric space decoder if prototypes have different length than embedding
        self.metric_space_decoder: Optional[torch.nn.Linear] = None
        if prototype_size != embeddings_size:
            self.metric_space_decoder = torch.nn.Linear(embeddings_size, prototype_size)
            torch.nn.init.xavier_uniform_(self.metric_space_decoder.weight)

        # create initial prototypes for all classes (all initial prototypes are a vector of all 1s)
        self.prototype_vectors = torch.nn.Parameter(torch.ones(num_prototypes, prototype_size), requires_grad=True)

        # if set, create initial prototypes from normal distribution
        if normal_distributed_initial_prototypes:
            self.prototype_vectors = torch.nn.Parameter(torch.normal(torch.zeros(num_prototypes, prototype_size)))

        # if set, use a radius
        self.prototype_radii: Optional[torch.nn.Parameter] = None
        if use_radius:
            self.prototype_radii = torch.nn.Parameter(torch.ones(num_prototypes), requires_grad=True)

        self.min_radius = min_radius
        self.learning_mode = learning_mode

        assert (unlabeled_idx is None) == (
            unlabeled_distance is None
        ), "'unlabeled_idx' and 'unlabeled_distance' should either both be set or both not be set."

        self.unlabeled_idx = unlabeled_idx
        self.unlabeled_distance = unlabeled_distance

        self._distance_function = distance_function

        self.distance: Optional[torch.nn.Module] = None
        if distance_function.lower() == "hyperbolic":
            self.distance = HyperbolicDistance()
        elif distance_function.lower() == "cosine":
            self.distance = CosineDistance()
        elif distance_function.lower() == "logit_cosine":
            self.distance = LogitCosineDistance()
        elif distance_function.lower() == "euclidean":
            self.distance = EuclideanDistance()
        elif distance_function.lower() == "dot_product":
            self.distance = NegativeScaledDotProduct()
        else:
            raise KeyError(f"Distance function {distance_function} not found.")

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    @property
    def num_prototypes(self):
        return self.prototype_vectors.size(0)

    def forward(self, embedded):
        if self.learning_mode == "learn_only_map_and_prototypes":
            embedded = embedded.detach()

        # decode embeddings into prototype space
        encoded = self.metric_space_decoder(embedded) if self.metric_space_decoder is not None else embedded

        prot = self.prototype_vectors
        radii = self.prototype_radii

        if self.learning_mode == "learn_only_prototypes":
            encoded = encoded.detach()

        if self.learning_mode == "learn_only_embeddings_and_map":
            prot = prot.detach()

            if radii is not None:
                radii = radii.detach()

        distance = self.distance(encoded, prot)

        if radii is not None:
            distance /= self.min_radius + torch.nn.functional.softplus(radii)

        # if unlabeled distance is set, mask out loss to unlabeled class prototype
        if self.unlabeled_distance:
            distance[..., self.unlabeled_idx] = self.unlabeled_distance

        scores = -distance

        return scores


class LabelVerbalizerDecoder(torch.nn.Module):
    """A class for decoding labels using the idea of siamese networks / bi-encoders. This can be used for all classification tasks in flair.

    Args:
        label_encoder (flair.embeddings.TokenEmbeddings):
            The label encoder used to encode the labels into an embedding.
        label_dictionary (flair.data.Dictionary):
            The label dictionary containing the mapping between labels and indices.

    Attributes:
        label_encoder (flair.embeddings.TokenEmbeddings):
            The label encoder used to encode the labels into an embedding.
        label_dictionary (flair.data.Dictionary):
            The label dictionary containing the mapping between labels and indices.

    Methods:
        forward(self, label_embeddings: torch.Tensor, context_embeddings: torch.Tensor) -> torch.Tensor:
            Takes the label embeddings and context embeddings as input and returns a tensor of label scores.

    Examples:
        label_dictionary = corpus.make_label_dictionary("ner")
        label_encoder = TransformerWordEmbeddings('bert-base-ucnased')
        label_verbalizer_decoder = LabelVerbalizerDecoder(label_encoder, label_dictionary)
    """

    def __init__(self, label_embedding: Embeddings, label_dictionary: Dictionary):
        super().__init__()
        self.label_embedding = label_embedding
        self.verbalized_labels: List[Sentence] = self.verbalize_labels(label_dictionary)
        self.to(flair.device)

    @staticmethod
    def verbalize_labels(label_dictionary: Dictionary) -> List[Sentence]:
        """Takes a label dictionary and returns a list of sentences with verbalized labels.

        Args:
            label_dictionary (flair.data.Dictionary): The label dictionary to verbalize.

        Returns:
            A list of sentences with verbalized labels.

        Examples:
            label_dictionary = corpus.make_label_dictionary("ner")
            verbalized_labels = LabelVerbalizerDecoder.verbalize_labels(label_dictionary)
            print(verbalized_labels)
            [Sentence: "begin person", Sentence: "inside person", Sentence: "end person", Sentence: "single org", ...]
        """
        verbalized_labels = []
        for byte_label, idx in label_dictionary.item2idx.items():
            str_label = byte_label.decode("utf-8")
            if label_dictionary.span_labels:
                if str_label == "O":
                    verbalized_labels.append("outside")
                elif str_label.startswith("B-"):
                    verbalized_labels.append("begin " + str_label.split("-")[1])
                elif str_label.startswith("I-"):
                    verbalized_labels.append("inside " + str_label.split("-")[1])
                elif str_label.startswith("E-"):
                    verbalized_labels.append("ending " + str_label.split("-")[1])
                elif str_label.startswith("S-"):
                    verbalized_labels.append("single " + str_label.split("-")[1])
            else:
                verbalized_labels.append(str_label)
        return list(map(Sentence, verbalized_labels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the label verbalizer decoder.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            The scores of the decoder.

        Raises:
            RuntimeError: If an unknown decoding type is specified.
        """
        if self.training or not self.label_embedding._everything_embedded(self.verbalized_labels):
            self.label_embedding.embed(self.verbalized_labels)

        label_tensor = torch.stack([label.get_embedding() for label in self.verbalized_labels])

        if self.training:
            store_embeddings(self.verbalized_labels, "none")

        scores = torch.mm(inputs, label_tensor.T)

        return scores


class WikipediaLabelVerbalizerDecoder(torch.nn.Module):
    """A class for appending Wikidata descriptions to Wikipedia URL target labels in Named Entity Linking tasks.
    Embedding all, then simple matrix multiplication.

    Args:
        label_encoder (flair.embeddings.TokenEmbeddings):
            The label encoder used to encode the labels into an embedding.
        label_dictionary:
            The label dictionary, so the Wikipedia Labels
        verbalizations (Optional):
            A dictionary that already includes the (or some of the) verbalizations of labels. Labels as keys, verbalizations as values.
            E.g. {"United_States": "United States democracy state republic country nation",
                  "Netherlands": "Netherlands country constituent state territory country of the Kingdom of the Netherlands
        ...

    Attributes:
        ...

    Methods:
        forward(self, label_embeddings: torch.Tensor, context_embeddings: torch.Tensor) -> torch.Tensor:
            ...

    Examples:
        ...
    """

    def __init__(self, label_embedding: Embeddings,
                 label_dictionary: Dictionary,
                 requires_masking: bool,
                 fast_inference: bool = True,
                 fast_inference_save_path: str = None,
                 decoder_hidden_size: int = 768,
                 inputs_size: int = 768,
                 num_negatives: int = 128,
                 verbalizations: Optional[Dict[str, str]] = None,
                 candidates = None):

        super().__init__()
        self.label_embedding = label_embedding
        if verbalizations:
            self.verbalizations = verbalizations
        else:
            self.verbalizations = {}
        self.label_dictionary = label_dictionary
        self.verbalized_labels: List[Sentence] = self.verbalize_labels(self, label_dictionary=self.label_dictionary)
        self.candidates = candidates

        self.requires_masking = requires_masking
        self.num_negatives = num_negatives

        self.fast_inference = fast_inference
        self.fast_inference_save_path = fast_inference_save_path
        if self.fast_inference_save_path and not Path(self.fast_inference_save_path).exists():
            Path(self.fast_inference_save_path).mkdir(parents=True, exist_ok=True)

        if self.fast_inference:
            assert self.fast_inference_save_path, "Need a path to save label embeddings!"

        self.is_first_batch_in_evaluation = True

        # if text encoding end label encoding do not have the same size (e.g. because EntityLinker uses first-last-concatenation):
        if inputs_size != label_embedding.embedding_length:
            self.labels_hidden = torch.nn.Linear(self.label_embedding.embedding_length, decoder_hidden_size)
            self.inputs_hidden = torch.nn.Linear(inputs_size, decoder_hidden_size)
            torch.nn.init.xavier_uniform_(self.labels_hidden.weight)
            torch.nn.init.xavier_uniform_(self.inputs_hidden.weight)
        else:
            self.labels_hidden = None
            self.inputs_hidden = None


        self.to(flair.device)

    @staticmethod
    def verbalize_labels(self, label_dictionary: Dictionary) -> List[Sentence]:

        verbalized_labels = []
        for byte_label, idx in label_dictionary.item2idx.items():
            print("Verbalizing", idx, byte_label, "...")
            str_label = byte_label.decode("utf-8")

            if label_dictionary.span_labels:
                if str_label not in ["O", "<unk>"]:
                    bio = str_label.split("-")[0]
                    entity = str_label[2:]
                    if bio == "B":
                        bio_verbalized = "begin "
                    elif bio == "I":
                        bio_verbalized = "inside "
                    elif bio == "E":
                        bio_verbalized = "ending "
                    elif bio == "S":
                        bio_verbalized = "single"
                else:
                    bio_verbalized = ""
                    entity = str_label

            else:
                entity = str_label

            if entity == "O":
                verbalized = "outside"
            elif entity == "<unk>":
                verbalized = "unknown"
            else:
                verbalized = self.verbalizations.get(entity, None)
                if not verbalized:
                    wikipedia_label = entity
                    #print(wikipedia_label)
                    wikidata_info = wikidata_NER_mapping.pipeline(wikipedia_label, method ="strict")
                    wikidata_classes = wikidata_info["class_names"]
                    wikidata_title = wikidata_info["wikidata_title"]
                    verbalized = wikidata_title + ", " + ", ".join(wikidata_classes)
                    self.verbalizations[entity] = verbalized

            if label_dictionary.span_labels:
                verbalized = bio_verbalized + verbalized

            verbalized_labels.append(verbalized)

        print(f"--- Created verbalized labels for {len(verbalized_labels)} labels")
        return list(map(Sentence, verbalized_labels))

    def return_verbalizations_dict(self):
        return self.verbalizations

    def remove_verbalizations_from_memory(self):
        del self.verbalizations

    def embed_labels_batchwise_and_save(self, batch_size = 128, mode= "embed"):
        assert mode in ["embed", "load_from_file"], "mode should be either `embed` or `load_from_file"

        if mode == "embed":
            label_embeddings = []
            indices = [idx for idx in range(len(self.verbalized_labels))]
            label_names = [byte_label.decode("utf-8") for byte_label, idx in self.label_dictionary.item2idx.items() ]

            for i in range(0, len(indices), batch_size):
                #indices_batch = indices[i:i+batch_size]
                labels_batch = self.verbalized_labels[i:i+batch_size]

                self.label_embedding.embed(labels_batch)
                label_embeddings.extend([label.get_embedding() for label in labels_batch])

                for s in labels_batch:
                    s.clear_embeddings()

            label_embeddings_tensor = torch.stack(label_embeddings)

            if self.fast_inference_save_path:
                label_tensor_path = self.fast_inference_save_path + "/label_embeddings.npy"
                np.save(label_tensor_path, label_embeddings_tensor.cpu().numpy())

                # save as tsv as well, for enabling visualizations
                label_tensor_tsv_path = self.fast_inference_save_path + "/label_embeddings.tsv"
                np.savetxt(label_tensor_tsv_path, label_embeddings_tensor.cpu().numpy(), delimiter='\t', fmt='%.8f')

                with open(self.fast_inference_save_path + "/label_embeddings_names.txt", 'w') as file:
                    for item in label_names:
                        file.write(item + '\n')


        elif mode == "load_from_file":
            label_tensor_path = self.fast_inference_save_path + "/label_embeddings.npy"
            label_embeddings_tensor = torch.from_numpy(np.load(label_tensor_path)).to(flair.device)

        else:
            raise ValueError

        return label_embeddings_tensor

    def embedding_sublist(self, labels) -> List[Sentence]:
        unique_entries = set(labels)

        # Randomly sample entries from the larger list
        while len(unique_entries) < self.num_negatives:
            entry = random.choice(range(len(self.verbalized_labels)))
            unique_entries.add(entry)

        return [self.verbalized_labels[idx] for idx in unique_entries], unique_entries

    def get_candidates_of_batch(self, data_points) -> List[Sentence]:
        unique_entries = set()
        unique_entries.add(self.label_dictionary.item2idx.get("<unk>".encode("utf-8")))

        for d in data_points:
            candidates = self.candidates.get_candidates(d.text)
            for c in candidates:
                idx = self.label_dictionary.item2idx.get(c.encode("utf-8"), None)
                if idx:
                    unique_entries.add(idx)

        return [self.verbalized_labels[idx] for idx in unique_entries], unique_entries

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor = None, data_points = None) -> torch.Tensor:

        if self.training:
            self.is_first_batch_in_evaluation = True # set this to assure new embedding of labels in the following evaluation phase

            if self.requires_masking:  # during training, only embed the labels that are in the batch (optionally also some additional negatives)
                labels_to_include = labels.cpu().numpy().tolist()
                reduced_labels, reduced_indices = self.embedding_sublist(labels_to_include)

            else:
                reduced_labels = self.verbalized_labels
                reduced_indices = [idx for idx in range(len(self.verbalized_labels))]

        else:
            # if using candidates, only the candidates in the batch need embedding, even in prediction. The rest gets masked in decoding anyway
            if self.candidates:
                reduced_labels, reduced_indices = self.get_candidates_of_batch(data_points=data_points)

            # embeds everything, but only once and batch_wise, saves as numpy array, reads in all following batches of same eval phase
            elif self.fast_inference:
                # if it's the first batch right after training, embed all and save!
                if self.is_first_batch_in_evaluation:
                    label_tensor = self.embed_labels_batchwise_and_save(mode = "embed")
                    self.is_first_batch_in_evaluation = False
                # else: read in the embeddings
                else:
                    try:
                        label_tensor = self.embed_labels_batchwise_and_save(mode = "load_from_file")
                    except:
                        label_tensor = self.embed_labels_batchwise_and_save(mode = "embed") # fallback, just in case

                reduced_labels = None
                reduced_indices = None

            else:
                reduced_labels = self.verbalized_labels
                reduced_indices = [idx for idx in range(len(self.verbalized_labels))]

        if reduced_labels and reduced_indices:
            self.label_embedding.embed(reduced_labels)
            label_tensor = torch.stack([label.get_embedding() for label in reduced_labels])

        if self.labels_hidden:
            label_tensor = self.labels_hidden(label_tensor)

        if self.inputs_hidden:
            inputs = self.inputs_hidden(inputs)

        store_embeddings(self.verbalized_labels, "none")
        # if self.training:
        #     store_embeddings(self.verbalized_labels, "none")
        # else:
        #     store_embeddings(self.verbalized_labels, "none") # todo is that a problem?

        scores = torch.mm(inputs, label_tensor.T)

        if reduced_labels and reduced_indices:
            all_scores = torch.zeros(scores.shape[0], len(self.verbalized_labels), device=flair.device)
            all_scores[:, torch.LongTensor(list(reduced_indices))] = scores

        else:
            all_scores = scores

        return all_scores

