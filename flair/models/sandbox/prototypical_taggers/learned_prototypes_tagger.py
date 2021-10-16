from typing import Optional, Union, List

import torch
from torch.nn.parameter import Parameter
from tqdm import tqdm

import flair
from flair.data import Sentence, Dictionary, Dataset, Label
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings
from flair.nn import Classifier
from flair.training_utils import store_embeddings
from .distance import EuclideanDistance, HyperbolicDistance


class LearnedPrototypesTagger(Classifier):
    def __init__(self,
                 embeddings: TokenEmbeddings,
                 tag_dictionary: Dictionary, tag_type: str,
                 prototype_size: int = 100,
                 unlabeled_distance: Optional[int] = None,
                 hyperbolic: Optional[bool] = False,
                 learning_mode=0,
                 ):
        """
        Prototypical model to tag tokens in a sentence using an embedding and
        an euclidean or hyperbolic distance metric.

        :param train_data: Train data used to compute prototypes on model.eval().
        :param embeddings: Embedding for the sentences the tokens occur in.
        The embedding should contain information about the sentence
        (otherwise token tagging done this way becomes pointless).
        :param tag_type: The tag to predict.
        :param support_size: The size of the support set (over all classes).
        This number can be retrieved from the episodic sampler.
        :param hyperbolic: Whether to use euclidean or hyperbolic distance.
        :param embedding_to_metric_space: Funktion to apply after the embedding.
        """
        super().__init__()
        self.embeddings = embeddings

        self.tag_type = tag_type
        self.prototype_size = prototype_size

        # initialize the label dictionary
        self.prototype_labels: Dictionary = tag_dictionary

        # potantial handling of "out" tags
        self.unlabeled_idx = tag_dictionary.get_idx_for_item('O')
        self.unlabeled_distance = unlabeled_distance

        if self.prototype_size or self.prototype_size == embeddings.embedding_length:
            self.prototype_size = embeddings.embedding_length
            self.metric_space_decoder = None
        else:
            # map embeddings to prototype space
            self.metric_space_decoder = torch.nn.Linear(embeddings.embedding_length, self.prototype_size)

        # create initial prototypes for all classes
        self.prototype_vectors = Parameter(torch.normal(torch.zeros(len(self.prototype_labels), self.prototype_size)))

        self._hyperbolic = hyperbolic

        self.loss = torch.nn.CrossEntropyLoss()

        if hyperbolic:
            self.distance = HyperbolicDistance()
        else:
            self.distance = EuclideanDistance()

        self.learning_mode = learning_mode

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    def embed_sentences(self, sentences):
        names = self.embeddings.get_names()

        self.embeddings.embed(sentences)

        embedded = torch.stack([
            torch.cat(token.get_each_embedding(names))
            for sentence in sentences for token in sentence
        ], dim=0)

        return embedded

    def forward_loss(self, sentences):
        return self._calculate_loss(self.forward(sentences), sentences)

    def _calculate_loss(self, feature, sentences):
        true_class = torch.tensor(
            self.prototype_labels.get_idx_for_items([
                token.get_tag(self.tag_type).value
                for sentence in sentences for token in sentence
            ])).to(flair.device)

        return self.loss(feature, true_class)

    def forward(self, sentences):
        assert self.prototype_vectors is not None

        # get embeddings for data points
        embedded = self.embed_sentences(sentences)

        # decode embeddings into prototype space
        if self.metric_space_decoder is not None:
            encoded = self.metric_space_decoder(embedded)
        else:
            encoded = embedded

        prot = self.prototype_vectors

        # used in alternating training only
        if self.learning_mode == 1:
            self.learning_mode = -1
            encoded = encoded.detach()
        elif self.learning_mode == -1:
            self.learning_mode = 1
            prot = prot.detach()

        distance = self.distance(encoded, prot)

        # if unlabeled distance is set, mask out loss to unlabeled class prototype
        if self.unlabeled_distance:
            distance[..., self.unlabeled_idx] = self.unlabeled_distance

        return -distance

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            mini_batch_size=32,
            all_tag_prob: bool = False,
            verbose: bool = False,
            label_name: Optional[str] = None,
            return_loss=False,
            embedding_storage_mode="none"
    ):
        """
        Predict sequence tags for Named Entity Recognition task
        :param sentences: a Sentence or a List of Sentence
        :param mini_batch_size: size of the minibatch, usually bigger is more rapid but consume more memory,
        up to a point when it has no more effect.
        :param all_tag_prob: True to compute the score for each tag on each token,
        otherwise only the score of the best tag is returned
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if label_name is None:
            label_name = self.tag_type

        with torch.no_grad():
            if not sentences:
                return sentences

            # read Dataset into data loader (if list of sentences passed, make Dataset first)
            if not isinstance(sentences, Dataset):
                sentences = SentenceDataset(sentences)

            dataloader = DataLoader(sentences, batch_size=mini_batch_size, )

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            overall_loss = 0
            batch_no = 0
            for batch in dataloader:

                batch_no += 1

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {batch_no}")

                feature = self.forward(sentences)

                if return_loss:
                    overall_loss += self._calculate_loss(feature, sentences)

                tags, all_tags = self._obtain_labels(
                    feature=feature,
                    get_all_tags=all_tag_prob,
                )

                tokens = [
                    token
                    for sentence in sentences
                    for token in sentence
                ]

                for (token, tag) in zip(tokens, tags):
                    token.add_tag_label(label_name, tag)

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (token, token_all_tags) in zip(batch, all_tags):
                    token.add_tags_proba_dist(label_name, token_all_tags)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss / batch_no

    def _obtain_labels(
            self,
            feature: torch.Tensor,
            get_all_tags: bool,
    ) -> (List[List[Label]], List[List[List[Label]]]):
        """
        Returns a tuple of two lists:
         - The first list corresponds to the most likely `Label` per token in each sentence.
         - The second list contains a probability distribution over all `Labels` for each token
           in a sentence for all sentences.
        """
        tags = []
        all_tags = []

        softmax_batch = torch.nn.functional.softmax(feature, dim=-1)

        softmax_batch = softmax_batch.cpu()

        probs_batch, prediction_batch = torch.max(softmax_batch, dim=-1)

        for all_probs, prob, pred in zip(softmax_batch, probs_batch, prediction_batch):
            tags.append(
                Label(self.prototype_labels.get_item_for_index(pred), prob)
            )

            if get_all_tags:
                all_tags.append([
                    Label(
                        self.prototype_labels.get_item_for_index(idx), idx_prob
                    )
                    for idx, idx_prob in enumerate(all_probs)
                ])

        return tags, all_tags

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "tag_type": self.tag_type,
            "hyperbolic": self._hyperbolic,
            "prototype_labels": self.prototype_labels,
            "prototype_vectors": self.prototype_vectors,
            "unlabeled_distance": self.unlabeled_distance,
            "prototype_size": self.prototype_size,
            "learning_mode": self.learning_mode
        }

        if self.metric_space_decoder:
            model_state['metric_space_decoder'] = self.metric_space_decoder

        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        model = LearnedPrototypesTagger(
            embeddings=state["embeddings"],
            tag_type=state["tag_type"],
            tag_dictionary=state["prototype_labels"],
            hyperbolic=state["hyperbolic"],
            prototype_size=state["prototype_size"],
            unlabeled_distance=state["unlabeled_distance"],
            learning_mode=state.get("learning_mode", None),
        )
        model.load_state_dict(state["state_dict"])
        model.prototype_labels = state["prototype_labels"]
        model.prototype_vectors = state["prototype_vectors"]

        if 'metric_space_decoder' in state:
            model.metric_space_decoder = state['metric_space_decoder']
        return model

    @property
    def label_type(self):
        return self.tag_type
