import logging
from typing import Callable, Dict, List, Set, Optional, Tuple, Union

import torch

import flair.embeddings
import flair.nn
from flair.training_utils import store_embeddings
from flair.data import Dictionary, Sentence, Span, DT, DT2
from abc import abstractmethod
import re
from collections import defaultdict

log = logging.getLogger("flair")


class CandidateGenerationMethod:
    """Abstract base class for methods that, given a mention,
    generate a set of candidates, so that the EntityLinker only
    scores among these candidates and not all entities
    """

    @abstractmethod
    def get_candidates(self, mentions: List[str]) -> Set[str]:
        """Given a list of entity mentions this methods returns a constrained set of entity
        candidates for the mentions"""
        raise NotImplementedError


class EntityLinker(flair.nn.DefaultClassifier[Sentence, Span]):
    """
    Entity Linking Model
    The model expects text/sentences with annotated entity mentions and predicts entities to these mentions.
    To this end a word embedding is used to embed the sentences and the embedding of the entity mention goes through a linear layer to get the actual class label.
    The model is able to predict '<unk>' for entity mentions that the model can not confidently match to any of the known labels.
    """

    def __init__(
        self,
        embeddings: flair.embeddings.TokenEmbeddings,
        label_dictionary: Dictionary,
        pooling_operation: str = "first_last",
        label_type: str = "nel",
        candidate_generation_method: Optional[CandidateGenerationMethod] = None,
        **classifierargs,
    ):
        """
        Initializes an EntityLinker
        :param embeddings: embeddings used to embed the words/sentences
        :param label_dictionary: dictionary that gives ids to all classes. Should contain <unk>
        :param pooling_operation: either 'average', 'first', 'last' or 'first&last'. Specifies the way of how text representations of entity mentions (with more than one word) are handled.
        E.g. 'average' means that as text representation we take the average of the embeddings of the words in the mention. 'first&last' concatenates
        the embedding of the first and the embedding of the last word.
        :param label_type: name of the label you use.
        """

        super(EntityLinker, self).__init__(
            embeddings=embeddings,
            label_dictionary=label_dictionary,
            final_embedding_size=embeddings.embedding_length * 2
            if pooling_operation == "first_last"
            else embeddings.embedding_length,
            **classifierargs,
        )

        self.pooling_operation = pooling_operation
        self._label_type = label_type

        cases: Dict[str, Callable[[Span, List[str]], torch.Tensor]] = {
            "average": self.emb_mean,
            "first": self.emb_first,
            "last": self.emb_last,
            "first_last": self.emb_firstAndLast,
        }

        if pooling_operation not in cases:
            raise KeyError('pooling_operation has to be one of "average", "first", "last" or "first_last"')

        self.aggregated_embedding = cases[pooling_operation]

        self.candidate_generation_method = candidate_generation_method

        self.to(flair.device)

    def emb_first(self, span: Span, embedding_names):
        return span.tokens[0].get_embedding(embedding_names)

    def emb_last(self, span: Span, embedding_names):
        return span.tokens[-1].get_embedding(embedding_names)

    def emb_firstAndLast(self, span: Span, embedding_names):
        return torch.cat(
            (span.tokens[0].get_embedding(embedding_names), span.tokens[-1].get_embedding(embedding_names)), 0
        )

    def emb_mean(self, span, embedding_names):
        return torch.mean(torch.cat([token.get_embedding(embedding_names) for token in span], 0), 0)

    def _get_data_points_from_sentence(self, sentence: Sentence) -> List[Span]:
        return sentence.get_spans(self.label_type)

    def _filter_data_point(self, data_point: Sentence) -> bool:
        return bool(data_point.get_labels(self.label_type))

    def _get_embedding_for_data_point(self, prediction_data_point: Span) -> torch.Tensor:
        return self.aggregated_embedding(prediction_data_point, self.embeddings.get_names())

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "word_embeddings": self.embeddings,
            "label_type": self.label_type,
            "label_dictionary": self.label_dictionary,
            "pooling_operation": self.pooling_operation,
            "loss_weights": self.weight_dict,
        }
        return model_state

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:

            eval_line = f"\n{datapoint.to_original_text()}\n"

            for span in datapoint.get_spans(gold_label_type):
                symbol = "✓" if span.get_label(gold_label_type).value == span.get_label("predicted").value else "❌"
                eval_line += (
                    f' - "{span.text}" / {span.get_label(gold_label_type).value}'
                    f' --> {span.get_label("predicted").value} ({symbol})\n'
                )

            lines.append(eval_line)
        return lines

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):

        # remap state dict for models serialized with Flair <= 0.11.3
        import re

        state_dict = state["state_dict"]
        for key in list(state_dict.keys()):
            state_dict[re.sub("^word_embeddings\\.", "embeddings.", key)] = state_dict.pop(key)

        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("word_embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            pooling_operation=state.get("pooling_operation"),
            loss_weights=state.get("loss_weights", {"<unk>": 0.3}),
            **kwargs,
        )

    @property
    def label_type(self):
        return self._label_type

    def _mask_scores_with_candidates(self, scores: torch.Tensor, data_points: List[DT2]):

        # get the candidates
        mentions = [span.text for span in data_points]
        candidate_set = self.candidate_generation_method.get_candidates(mentions)

        if candidate_set:
            indices_of_candidates = [self.label_dictionary.get_idx_for_item(candidate) for candidate in candidate_set]
            # mask out (set to -inf) logits that are not in the proposed candidate set
            masked_scores = -torch.inf * torch.ones(scores.size(), requires_grad=True, device=flair.device)
            masked_scores[:, indices_of_candidates] = scores[:, indices_of_candidates]
            return masked_scores

        return scores

    def forward_loss(self, sentences: List[DT]) -> Tuple[torch.Tensor, int]:

        # make a forward pass to produce embedded data points and labels
        sentences = [sentence for sentence in sentences if self._filter_data_point(sentence)]

        # get the data points for which to predict labels
        data_points = self._get_data_points_for_batch(sentences)
        if len(data_points) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # get their gold labels as a tensor
        label_tensor = self._prepare_label_tensor(data_points)
        if label_tensor.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # pass data points through network to get encoded data point tensor
        data_point_tensor = self._encode_data_points(sentences, data_points)

        # decode
        scores = self.decoder(data_point_tensor)

        if self.candidate_generation_method:
            self._mask_scores_with_candidates(scores, data_points)

        # calculate the loss
        return self._calculate_loss(scores, label_tensor)

    def predict(
        self,
        sentences: Union[List[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ):
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.  # noqa: E501
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param return_probabilities_for_all_classes : return probabilities for all classes instead of only best predicted  # noqa: E501
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted  # noqa: E501
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.  # noqa: E501
        'gpu' to store embeddings in GPU memory.
        """
        if label_name is None:
            label_name = self.label_type if self.label_type is not None else "label"

        with torch.no_grad():
            if not sentences:
                return sentences

            if not isinstance(sentences, list):
                sentences = [sentences]

            reordered_sentences = self._sort_data(sentences)

            if len(reordered_sentences) == 0:
                return sentences

            if len(reordered_sentences) > mini_batch_size:
                batches: Union[DataLoader, List[List[DT]]] = DataLoader(
                    dataset=FlairDatapointDataset(reordered_sentences),
                    batch_size=mini_batch_size,
                )
                # progress bar for verbosity
                if verbose:
                    progress_bar = tqdm(batches)
                    progress_bar.set_description("Batch inference")
                    batches = progress_bar
            else:
                batches = [reordered_sentences]

            overall_loss = torch.zeros(1, device=flair.device)
            label_count = 0
            for batch in batches:

                # filter data points in batch
                batch = [dp for dp in batch if self._filter_data_point(dp)]

                # stop if all sentences are empty
                if not batch:
                    continue

                data_points = self._get_data_points_for_batch(batch)

                if not data_points:
                    continue

                # pass data points through network and decode
                data_point_tensor = self._encode_data_points(batch, data_points)
                scores = self.decoder(data_point_tensor)

                if self.candidate_generation_method:
                    scores = self._mask_scores_with_candidates(scores, data_points)

                # if anything could possibly be predicted
                if len(data_points) > 0:
                    # remove previously predicted labels of this type
                    for sentence in data_points:
                        sentence.remove_labels(label_name)

                    if return_loss:
                        gold_labels = self._prepare_label_tensor(data_points)
                        overall_loss += self._calculate_loss(scores, gold_labels)[0]
                        label_count += len(data_points)

                    softmax = torch.nn.functional.softmax(scores, dim=-1)

                    if return_probabilities_for_all_classes:
                        n_labels = softmax.size(1)
                        for s_idx, data_point in enumerate(data_points):
                            for l_idx in range(n_labels):
                                label_value = self.label_dictionary.get_item_for_index(l_idx)
                                if label_value == "O":
                                    continue
                                label_score = softmax[s_idx, l_idx].item()
                                data_point.add_label(typename=label_name, value=label_value, score=label_score)
                    else:
                        conf, idx = torch.max(softmax, dim=-1)
                        for data_point, c, i in zip(data_points, conf, idx):
                            label_value = self.label_dictionary.get_item_for_index(i.item())
                            if label_value == "O":
                                continue
                            data_point.add_label(typename=label_name, value=label_value, score=c.item())

                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count


class SimpleCandidateGenerator(CandidateGenerationMethod):
    """Straight forward candidate generator using mention-candidate lists,
    i.e. we assume to have a dictionary that stores mentions as keys and
    a list of possible candidates as values.
    These lists typically originate from large scale annotated corpora like Wikipedia.
    """

    def __init__(self, candidate_dict: Dict):

        # do not use defaultdict for the original candidate dict
        # otherwise the get_candidates fct does not work properly (never throws KeyError)
        self.candidate_dict = candidate_dict

        self.punc_remover = re.compile(r"[\W]+")

        # to improve the recall of the candidate lists we add a lower cased and a further reduced version of each
        # mention to the mention set, note that simplifying mentions may join some candidate lists
        # because the simplified mentions of prior different mentions may coincide
        self.simpler_mentions_candidate_dict = defaultdict(set)
        self.even_more_simpler_mentions_candidate_dict = defaultdict(set)
        for mention in candidate_dict:
            # create mention without blanks and lower cased
            simplified_mention = mention.replace(" ", "").lower()
            self.simpler_mentions_candidate_dict[simplified_mention].update(self.candidate_dict[mention])
            # create further reduced mention
            more_simplified_mention = self.punc_remover.sub("", mention.lower())
            self.even_more_simpler_mentions_candidate_dict[more_simplified_mention].update(self.candidate_dict[mention])

    def get_candidates(self, mentions: List[str]) -> Set[str]:
        candidates_for_all_mentions = set()
        for mention in mentions:
            candidates = set()
            try:
                candidates.update(self.candidate_dict[mention])
            except KeyError:
                candidates = self.simpler_mentions_candidate_dict[mention.lower().replace(" ", "")]
                if not candidates:
                    candidates = self.even_more_simpler_mentions_candidate_dict[
                        self.punc_remover.sub("", mention.lower())
                    ]
            candidates_for_all_mentions.update(candidates)

        return candidates_for_all_mentions


class EntityDecoder(torch.nn.Module):
    """
    Simple linear decoder with two linear layers. Can be used to reduce (or choose) the dimension
    of the final linear layer ('entity embeddings'). Since one might deals with a huge entity set, chooseing a smaller
    dimension can help to reduce memory usage.
    """

    def __init__(self, entity_embedding_size: int, mention_embedding_size: int, number_entities: int):
        super().__init__()

        self.mention_to_entity = torch.nn.Linear(
            mention_embedding_size, entity_embedding_size
        )  # project mention embedding to entity embedding space

        self.entity_embeddings = torch.nn.Linear(
            entity_embedding_size, number_entities, bias=False
        )  # each entity is represented by a vector

    def forward(self, mention_embeddings):

        # project mentions in entity representation
        projected_mention_embeddings = self.mention_to_entity(mention_embeddings)

        # compute scores
        logits = self.entity_embeddings(projected_mention_embeddings)

        return logits
