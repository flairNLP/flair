import logging
from typing import Callable, Dict, List, Tuple, Set, Union, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

import flair.embeddings
import flair.nn
from flair.data import Dictionary, Sentence, Span, DT
from flair.training_utils import store_embeddings
import re
from collections import defaultdict

from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

from abc import abstractmethod

log = logging.getLogger("flair")


class GenerativeEntityLinker(flair.nn.DefaultClassifier[Sentence, Span]):
    """
    Entity Linking Model
    The model expects text/sentences with annotated entity mentions and predicts entities to these mentions.
    To this end a word embedding is used to embed the sentences and the embedding of the entity mention goes through a linear layer to get the actual class label.
    The model is able to predict '<unk>' for entity mentions that the model can not confidently match to any of the known labels.
    """

    def __init__(
        self,
        embeddings: flair.embeddings.TokenEmbeddings,
        decoder,  #: GenerativeDecoder,
        pooling_operation: str = "first_last",
        label_type: str = "nel",
        label_dictionary=None,
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

        super(GenerativeEntityLinker, self).__init__(
            embeddings=embeddings,
            label_dictionary=None,
            final_embedding_size=embeddings.embedding_length * 2
            if pooling_operation == "first_last"
            else embeddings.embedding_length,
            decoder=decoder,
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

        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

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

    def forward_loss(self, sentences: List[DT]) -> Tuple[torch.Tensor, int]:

        # make a forward pass to produce embedded data points and labels
        sentences = [sentence for sentence in sentences if self._filter_data_point(sentence)]

        # get the data points for which to predict labels
        data_points = self._get_data_points_for_batch(sentences)
        if len(data_points) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # pass data points through network to get encoded data point tensor
        data_point_tensor = self._encode_data_points(sentences, data_points)

        # decode
        logits, attention_mask, labels_input_ids = self.decoder.teacher_forcing_decode(data_point_tensor, data_points)

        target_ids_for_loss = pad_sequence(labels_input_ids, batch_first=True, padding_value=-1)

        # logits are in the form (#mentions, max_seq_length, gpt2vocab size)
        # we need the form (#mentions, gpt2vocab size, max_seq_length) for the loss
        logits = torch.permute(logits, (0, 2, 1))

        # calculate the loss
        return self.cross_entropy(logits, target_ids_for_loss), len(data_points)

    def _calculate_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        return self.loss_function(scores, labels), labels.size(0)

    def predict(
        self,
        sentences: Union[List[DT], DT],
        mini_batch_size: int = 1,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
        prefix_trie=None,
        max_num_steps=40,
        beam_size=5,
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

            # filter and sort sentences
            reordered_sentences = self._sort_data(sentences)
            reordered_sentences = [sentence for sentence in reordered_sentences if self._filter_data_point(sentence)]

            if len(reordered_sentences) == 0 and return_loss:
                return torch.zeros(1, device=flair.device), 0

            overall_loss = torch.zeros(1, device=flair.device)
            label_count = 0
            for sentence in reordered_sentences:

                data_points = self._get_data_points_from_sentence(sentence)

                if not data_points:
                    continue

                # pass data points through network and decode
                data_point_tensor = self._encode_data_points([sentence], data_points)

                # generative inference for each mention separately
                outputs = []
                for mention_embedding in data_point_tensor:
                    output_string = self.decoder.generative_inference(
                        embedded_mention=mention_embedding.unsqueeze(0),
                        prefix_trie=prefix_trie,
                        beam_size=beam_size,
                        max_num_steps=max_num_steps,
                    )
                    outputs.append(output_string)

                # if anything could possibly be predicted
                if len(data_points) > 0:
                    # remove previously predicted labels of this type
                    sentence.remove_labels(label_name)

                    if return_loss:
                        gold_labels = [span.get_label("nel").value for span in data_points]
                        for gold_label, prediction in zip(gold_labels, outputs):
                            if gold_label != prediction:
                                overall_loss += 1
                        label_count += len(data_points)

                    for data_point, prediction in zip(data_points, outputs):
                        data_point.add_label(typename=label_name, value=prediction)

                store_embeddings([sentence], storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count


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


class GenerativeDecoder(torch.nn.Module):
    # TODO: add possibility to add other decoder (not only GPT2)
    def __init__(self, max_seq_length=40, embedding_dim=2 * 768, num_layers=4, label_type: str = "nel"):

        my_config = GPT2Config(
            n_positions=max_seq_length,  # maximum sequence length
            n_embd=embedding_dim,  # dim of embeddings and hidden states (2* encoder dim if I do pooling,
            # otherwise same as encoder dim)
            n_layer=num_layers,  # number of hidden layers
            # embd_pdrop=0, # for testing one can set the dropouts to 0
            # resid_pdrop=0,
            # attn_pdrop=0,
            # summary_first_dropout=0
        )
        super().__init__()

        self.transformer_decoder = GPT2Model(my_config)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.final_linear_layer = torch.nn.Linear(my_config.n_embd, my_config.vocab_size, bias=False)

        self.softmax = torch.nn.Softmax(dim=1)

        self.label_type = label_type

    def _get_token_ids(self, label: str):

        input_ids = self.tokenizer(label)["input_ids"]

        return torch.tensor(input_ids + [self.tokenizer.eos_token_id], device=flair.device)

    def teacher_forcing_decode(self, embedded_data_points, data_points):

        # num mentions, 1, mention_emb_dim
        hidden_states_encoder = embedded_data_points.unsqueeze(1)

        # now, for teacher forcing we need to add the gold labels to the input
        labels_input_ids = []
        label_embedding_vectors = []
        for span in data_points:
            label = span.get_label(self.label_type).value
            input_ids = self._get_token_ids(label)
            # it might happen that a label is too large for the tranformer decoder, i.e. exceeds the maximum sequence length
            if len(input_ids) > self.transformer_decoder.config.n_positions:
                print(
                    f"Warning: Label: {label} (length {len(input_ids)}) exceeds width of decoder (length {self.transformer_decoder.config.n_positions}).\n"
                    f"Label will be cut off and cannot correctly be generated."
                )
                input_ids = input_ids[: self.transformer_decoder.config.n_positions]
            labels_input_ids.append(input_ids)
            label_embedding_vectors.append(self.transformer_decoder.wte(input_ids[:-1]))

        # num mentions, max sequence length, mention_emb_dim
        padded_label_embedding_vectors = pad_sequence(label_embedding_vectors, batch_first=True)

        attention_mask = torch.tensor(
            [
                [1] * (vec.size()[0] + 1) + [0] * (padded_label_embedding_vectors.size()[1] - vec.size()[0])
                for vec in label_embedding_vectors
            ],
            device=flair.device,
        )

        decoder_input_vectors = torch.cat([hidden_states_encoder, padded_label_embedding_vectors], dim=1)

        output = self.transformer_decoder.forward(inputs_embeds=decoder_input_vectors, attention_mask=attention_mask)

        # now we compute the predictions
        logits = self.final_linear_layer(output.last_hidden_state)

        return logits, attention_mask, labels_input_ids

    def generative_inference(self, embedded_mention, max_num_steps=40, prefix_trie: Dict = None, beam_size=5):

        # past_key_values = None

        with torch.no_grad():
            # first step:
            output = self.transformer_decoder.forward(inputs_embeds=embedded_mention)

            # beams
            logits = self.final_linear_layer(output.last_hidden_state[-1, :])
            logits[self.tokenizer.eos_token_id] = -torch.inf  # exclude end_of_sentence token for first prediction

            if prefix_trie:
                sensible_continuations = list(prefix_trie.keys())
                # TODO: if sensible continuations has length 1 I do not need to contunue prediction but can directly recursively scroll through the tree
                masked_logits = -torch.inf * torch.ones(logits.size(), requires_grad=True, device=flair.device)
                masked_logits[sensible_continuations] = logits[sensible_continuations]

                # temporary fix: it may happen that there are less sensible conitnuations than the beam_size,
                # in this case we need to reduce the beam size
                beam_size = min(beam_size, len(sensible_continuations))

                output_probabilities = torch.nn.functional.softmax(masked_logits, dim=0)

                probabilities, indices = torch.topk(output_probabilities, beam_size, dim=0)

                beam_sequences = [
                    [[index.item()], torch.log(prob), prefix_trie[index.item()]]
                    for index, prob in zip(indices, probabilities)
                ]

            else:
                output_probabilities = torch.nn.functional.softmax(logits, dim=0)

                probabilities, indices = torch.topk(output_probabilities, beam_size, dim=0)

                beam_sequences = [[[index.item()], torch.log(prob)] for index, prob in zip(indices, probabilities)]

            final_candidates = []

            # create next input
            beam_mention_embedding = embedded_mention.repeat(
                beam_size, 1, 1
            )  # initial embedding repeated beam_size times
            indices_embeddings = self.transformer_decoder.wte(indices).unsqueeze(1)

            embedded_beams = torch.cat((beam_mention_embedding, indices_embeddings), dim=1)

            # next step
            # size: #beams, 2, hidden_dim
            output = self.transformer_decoder.forward(inputs_embeds=embedded_beams)

            for gen_step in range(1, max_num_steps - 1):

                # past_key_values = output.past_key_values

                logits = self.final_linear_layer(output.last_hidden_state[:, -1, :])

                if prefix_trie:
                    # for each sequence in the beam there is a particular subtree
                    masked_logits = -torch.inf * torch.ones(logits.size(), requires_grad=True, device=flair.device)
                    for i, beam in enumerate(beam_sequences):
                        sensible_continuations = list(beam[2].keys())
                        masked_logits[i, sensible_continuations] = logits[i, sensible_continuations]

                    output_probabilities = self.softmax(masked_logits)

                else:
                    # initial size: #beam_size, vocab_size
                    output_probabilities = self.softmax(logits)

                log_probs = torch.log(output_probabilities)

                scores, indices = torch.topk(log_probs, beam_size, dim=1)  # TODO: decreasing beam_size??!!

                all_expanded_candidates = []
                for i, seq in enumerate(beam_sequences):
                    for index, score in zip(indices[i, :], scores[i, :]):
                        if not score == -torch.inf and not torch.isnan(score):
                            if prefix_trie:
                                candidate = [seq[0] + [index.item()], seq[1] + score, seq[2][index.item()]]
                            else:
                                candidate = [seq[0] + [index.item()], seq[1] + score]
                            all_expanded_candidates.append(candidate)

                ordered_candidates = sorted(all_expanded_candidates, key=lambda tup: tup[1], reverse=True)

                beam_sequences = []
                new_indices = []
                for cand in ordered_candidates:
                    if cand[0][-1] == self.tokenizer.eos_token_id:  # end of sequence predicted
                        cand[1] = cand[1] / len(cand[0])
                        final_candidates.append(cand)
                    else:
                        beam_sequences.append(cand)
                        new_indices.append(cand[0])

                beam_sequences = beam_sequences[:beam_size]
                new_indices = new_indices[:beam_size]

                # TODO: when do we stop?
                # I fear that small sequences will be preferred
                # Especially if we use a prefix trie there will be many finished sequences early on
                # But luckily this only concerns the testing stage I will have enough possibilities to test, given a trained model
                # One possibility would be:
                # as soon as the final_list is large enough we stop and finish the beam_size times many unfisnished sequences in the beam greedily
                # i.e. until each single beam produces an eos token
                if len(final_candidates) >= beam_size**2 or not beam_sequences:
                    break

                # do the next decoding pass
                new_embedds = self.transformer_decoder.wte(torch.tensor(new_indices, device=flair.device))
                beam_mention_embedding = embedded_mention.repeat(len(new_indices), 1, 1)

                embedded_beams = torch.cat((beam_mention_embedding, new_embedds), dim=1)

                output = self.transformer_decoder.forward(inputs_embeds=embedded_beams)

        # add a sequence without eos-token to make sure at least one sequence is contained in final sequence list
        if beam_sequences:
            cand = beam_sequences[0]
            cand[1] = cand[1] / len(cand[0])
            final_candidates.append(cand)

        # TODO: Here I could add a greedy decoding for the remaining unfinished sequences in the beam

        ordered_final_candidates = sorted(final_candidates, key=lambda tup: tup[1], reverse=True)

        generated_sequence = ordered_final_candidates[0][0]
        if generated_sequence[-1] == self.tokenizer.eos_token_id:
            generated_sequence = generated_sequence[:-1]
        text_generated_sequence = [self.tokenizer.convert_ids_to_tokens(output_id) for output_id in generated_sequence]

        # transform ids to text output
        output_string = "".join(text_generated_sequence)

        return output_string


# TODO: Tree with nodes
"""
class TreeNode:
  def __init__(self, index):
    self.index = index
    self.children = []
"""


class PrefixTree(object):
    def __init__(self, sequences: List[List[int]]):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                PrefixTree._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.depth = self._compute_depth(self.trie_dict)

    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            PrefixTree._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    def __len__(self):
        return self.len

    def _compute_depth(self, trie_dict):

        if not trie_dict.keys():
            return 0
        else:
            return max([self._compute_depth(trie_dict[key]) for key in trie_dict]) + 1
