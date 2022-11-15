import logging
from typing import List, Union, Dict

import torch

import flair.embeddings
import flair.nn
from flair.data import Dictionary, Sentence
from flair.training_utils import  store_embeddings
import re

log = logging.getLogger("flair")


class EntityLinker(flair.nn.DefaultClassifier[Sentence]):
    """
    Entity Linking Model
    The model expects text/sentences with annotated entity mentions and predicts entities to these mentions.
    To this end a word embedding is used to embed the sentences and the embedding of the entity mention goes through a linear layer to get the actual class label.
    The model is able to predict '<unk>' for entity mentions that the model can not confidently match to any of the known labels.
    """

    def __init__(
        self,
        word_embeddings: flair.embeddings.TokenEmbeddings,
        label_dictionary: Dictionary,
        pooling_operation: str = "first_last",
        label_type: str = "nel",
        decoder = None,
        **classifierargs,
    ):
        """
        Initializes an EntityLinker
        :param word_embeddings: embeddings used to embed the words/sentences
        :param label_dictionary: dictionary that gives ids to all classes. Should contain <unk>
        :param pooling_operation: either 'average', 'first', 'last' or 'first&last'. Specifies the way of how text representations of entity mentions (with more than one word) are handled.
        E.g. 'average' means that as text representation we take the average of the embeddings of the words in the mention. 'first&last' concatenates
        the embedding of the first and the embedding of the last word.
        :param label_type: name of the label you use.
        """

        super(EntityLinker, self).__init__(
            label_dictionary=label_dictionary,
            final_embedding_size=word_embeddings.embedding_length * 2
            if pooling_operation == "first_last"
            else word_embeddings.embedding_length,
            decoder=decoder,
            **classifierargs,
        )

        self.word_embeddings = word_embeddings
        self.pooling_operation = pooling_operation
        self._label_type = label_type

        cases = {
            "average": self.emb_mean,
            "first": self.emb_first,
            "last": self.emb_last,
            "first_last": self.emb_firstAndLast,
        }

        if pooling_operation not in cases:
            raise KeyError('pooling_operation has to be one of "average", "first", "last" or "first_last"')

        self.aggregated_embedding = cases[pooling_operation]

        self.to(flair.device)
        #print(type(decoder))
        #if isinstance(decoder, GenerativeDecoder):

        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward_loss(self, sentences):# -> Tuple[torch.Tensor, int]:
        # make a forward pass to produce embedded data points and labels
        embedded_data_points, labels, data_points = self.forward_pass(sentences, for_prediction=True)  # type: ignore

        # no loss can be calculated if there are no labels
        if not any(labels):
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # use dropout
        # TODO: Where to add dropout, on embeddings of encoder or final embeddings of decoder
        # embedded_data_points = embedded_data_points.unsqueeze(1)
        # embedded_data_points = self.dropout(embedded_data_points)
        # embedded_data_points = self.locked_dropout(embedded_data_points)
        # embedded_data_points = self.word_dropout(embedded_data_points)
        # embedded_data_points = embedded_data_points.squeeze(1)

        # push embedded_data_points through decoder to get the scores
        mentions = [span.text for span in data_points]
        # logits, ids_of_candidates_from_batch = self.decoder(embedded_data_points, mentions)
        logits, attention_mask, labels_input_ids = self.decoder.decode(embedded_data_points, mentions, labels)

        target_ids_for_loss = pad_sequence(labels_input_ids, batch_first=True, padding_value=-1)

        # logits are in the form (#mentions, max_seq_length, gpt2vocab size)
        # we need the form (#mentions, gpt2vocab size, max_seq_length) for the loss
        logits = torch.permute(logits, (0,2,1))

        # calculate the loss
        return self.cross_entropy(logits, target_ids_for_loss), len(labels)

    """
    sentences: Union[List[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    """

    def predict(self,
                sentences,
                mini_batch_size: int = 1,
                return_probabilities_for_all_classes: bool = False,
                verbose: bool = False,
                label_name: str = None,
                return_loss=False,
                embedding_storage_mode="none",
                prefix_tree = None,
                beam_size=5):

        label_name='predicted'

        with torch.no_grad():
            if not sentences:
                return sentences

            if not isinstance(sentences, list):
                sentences = [sentences]

            overall_loss = torch.zeros(1, device=flair.device)
            label_count = 0

            for sentence in sentences:

                # stop if all sentences are empty
                if not sentence:
                    continue

                embedded_data_points, gold_labels, data_points = self.forward_pass(  # type: ignore
                    sentence, for_prediction=True
                )
                # if anything could possibly be predicted
                if len(data_points) > 0:
                    mentions = [span.text for span in data_points]

                    # I do inference for each mention separately
                    outputs = []
                    for mention_embedding in embedded_data_points:
                        output_string = self.decoder.generative_inference(embedded_mention=mention_embedding.unsqueeze(0), prefix_tree=prefix_tree, beam_size=beam_size)
                        outputs.append(output_string)

                    # remove previously predicted labels of this type
                    for data_point in data_points:
                        data_point.remove_labels(label_name)

                    if return_loss:
                        # overall_loss += self._calculate_loss(logits, gold_labels)[0]
                        # label_count += len(data_points)
                        overall_loss += 1
                        label_count += len(data_points)

                    for data_point, title in zip(data_points, outputs):

                        # label_value = self.label_dictionary.get_idx_for_item(title)
                        # if label_value == 0:
                        #     label_value='pred not a valid wikipedia title'
                        data_point.add_label(typename=label_name, value=title, score=1)

                store_embeddings(sentence, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count


    def emb_first(self, arg):
        return arg[0]

    def emb_last(self, arg):
        return arg[-1]

    def emb_firstAndLast(self, arg):
        return torch.cat((arg[0], arg[-1]), 0)

    def emb_mean(self, arg):
        return torch.mean(arg, 0)

    def forward_pass(
        self,
        sentences: Union[List[Sentence], Sentence],
        for_prediction: bool = False,
    ):

        if not isinstance(sentences, list):
            sentences = [sentences]

        # filter sentences with no candidates (no candidates means nothing can be linked anyway)
        filtered_sentences = []
        for sentence in sentences:
            if sentence.get_labels(self.label_type):
                filtered_sentences.append(sentence)

        # fields to return
        data_points = []
        span_labels = []
        embedded_mentions = None

        # embed sentences and send through prediction head
        if len(filtered_sentences) > 0:
            # embed all tokens
            self.word_embeddings.embed(filtered_sentences)

            embedding_names = self.word_embeddings.get_names()

            embedding_list = []
            # get the embeddings of the entity mentions
            for sentence in filtered_sentences:
                entities = sentence.get_spans(self.label_type)

                for entity in entities:

                    # get the label of the entity
                    span_labels.append([entity.get_label(self.label_type).value])

                    if self.pooling_operation == "first_last":
                        mention_emb = torch.cat(
                            (
                                entity.tokens[0].get_embedding(embedding_names),
                                entity.tokens[-1].get_embedding(embedding_names),
                            ),
                            0,
                        )
                    embedding_list.append(mention_emb.unsqueeze(0))

                if for_prediction:
                    data_points.extend(entities)
            if len(embedding_list) > 0:
                embedded_mentions = torch.cat(embedding_list, 0)

        if for_prediction:
            return embedded_mentions, span_labels, data_points

        return embedded_mentions, span_labels

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "word_embeddings": self.word_embeddings,
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
        return super()._init_model_with_state_dict(
            state,
            word_embeddings=state.get("word_embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            pooling_operation=state.get("pooling_operation"),
            loss_weights=state.get("loss_weights", {"<unk>": 0.3}),
            **kwargs,
        )

    @property
    def label_type(self):
        return self._label_type


class CandidateGenerationMethod():

    def __init__(self, candidate_dict):
        self.candidate_dict = candidate_dict

        # create a dictionary where blanks are removed
        # this leads to some distinct mentions being the same afterwards
        # thus we unify their lists
        # simpler_mentions_candidate_dict = {}
        # for mention in candidate_dict:
        #     # create mention without blanks
        #     simplified_mention=mention.replace(' ','').lower()
        #     # the simplified mention already occurred from another mention
        #     if simplified_mention in simpler_mentions_candidate_dict:
        #         candidates = set(simpler_mentions_candidate_dict[simplified_mention])
        #         candidates.update(candidate_dict[mention])
        #         simpler_mentions_candidate_dict[simplified_mention] = list(candidates)
        #     # its the first occurrence of the simplified mention
        #     else:
        #         simpler_mentions_candidate_dict[simplified_mention] = candidate_dict[mention]
        #
        # self.simpler_mentions_candidate_dict = simpler_mentions_candidate_dict

        self.simpler_mention_to_mentions = {}

        for mention in candidate_dict:
            # create mention without blanks
            simplified_mention=mention.replace(' ','').lower()
            # the simplified mention already occurred from another mention
            if simplified_mention in self.simpler_mention_to_mentions:
                self.simpler_mention_to_mentions[simplified_mention].add(mention)
            # its the first occurrence of the simplified mention
            else:
                self.simpler_mention_to_mentions[simplified_mention] = set(mention)


        self.punc_remover = re.compile(r"[\W]+")

        # even_more_simpler_mentions_candidate_dict = {}
        # for mention in candidate_dict:
        #     # create mention without blanks
        #     simplified_mention=self.punc_remover.sub("", mention.lower())
        #     # the simplified mention already occurred from another mention
        #     if simplified_mention in even_more_simpler_mentions_candidate_dict:
        #         candidates = set(even_more_simpler_mentions_candidate_dict[simplified_mention])
        #         candidates.update(candidate_dict[mention])
        #         even_more_simpler_mentions_candidate_dict[simplified_mention] = list(candidates)
        #     # its the first occurrence of the simplified mention
        #     else:
        #         even_more_simpler_mentions_candidate_dict[simplified_mention] = candidate_dict[mention]
        #
        # self.even_more_simpler_mentions_candidate_dict = even_more_simpler_mentions_candidate_dict

        self.even_simpler_mention_to_mentions = {}

        for mention in candidate_dict:
            # create mention without blanks
            simplified_mention = self.punc_remover.sub("", mention.lower())
            # the simplified mention already occurred from another mention
            if simplified_mention in self.even_simpler_mention_to_mentions:
                self.even_simpler_mention_to_mentions[simplified_mention].add(mention)
            # its the first occurrence of the simplified mention
            else:
                self.even_simpler_mention_to_mentions[simplified_mention] = set(mention)

    def get_candidates(self, mentions):
        candidate_set = set()
        for org_mention in mentions:
            try:
               candidate_set.update(self.candidate_dict[org_mention])
            except KeyError:
                # check if mention without blanks exists
                try:
                    mention_without_blanks = org_mention.replace(' ','').lower()
                    for mention in self.simpler_mention_to_mentions[mention_without_blanks]:
                        candidate_set.update(self.candidate_dict[mention])
                except KeyError:
                    # check if even more simplified mention exists
                    even_more_simplified_mention = self.punc_remover.sub("", org_mention.lower())
                    try:
                        for mention in self.even_simpler_mention_to_mentions[even_more_simplified_mention]:
                            candidate_set.update(self.candidate_dict[mention])
                    except KeyError:
                        #possible 'fixes': sample 15 random entities or sample candidates with string similarity (like 15 most similar)
                        pass
                        #print('Unknown mention: '+ mention)
        return candidate_set


class CandidateDecoder(torch.nn.Module):

    def __init__(self,
             entity_embedding_size: int,
             mention_embedding_size: int,
             entity_dictionary,
             candidate_generation_method,
             use_minus_infinity_instead_of_selection: bool = False
    ):
        super().__init__()

        self.use_minus_infinity_instead_of_selection = use_minus_infinity_instead_of_selection

        self.entity_dictionary = entity_dictionary # index for each entity (title or id)

        self.mention_to_entity = torch.nn.Linear(mention_embedding_size, entity_embedding_size) # project mention embedding to entity embedding space

        self.entity_embeddings = torch.nn.Linear(entity_embedding_size ,len(entity_dictionary), bias=False) # each entity is represented by a vector
        # TODO: more sophisticated way than random for initialization for entity vectors??

        self.candidate_generation_method = candidate_generation_method

    def forward(self,mention_embeddings, mentions):

        # create the restricted set of entities for which we compute the scores batch-wise
        scoring_entity_set = self.candidate_generation_method.get_candidates(mentions)

        # if no entity set is given, score over all entities
        if scoring_entity_set:
            indices_of_scoring_entities = [self.entity_dictionary.get_idx_for_item(entity) for entity in scoring_entity_set]

        # project mentions in entity representation
        projected_mention_embeddings = self.mention_to_entity(mention_embeddings)
        # compute scores
        logits = self.entity_embeddings(projected_mention_embeddings)
        # if not scoring over all entities we return the corresponding indices
        if scoring_entity_set:
            if not self.use_minus_infinity_instead_of_selection:
                return logits, indices_of_scoring_entities
            else:
                # mask out (set to -inf) logits that are not in the proposed candidate set
                masked_logits =-torch.inf*torch.ones(logits.size(), requires_grad=True,device=flair.device)
                masked_logits[:, indices_of_scoring_entities] = logits[:, indices_of_scoring_entities]
                return masked_logits, None
        else:
            return logits, None

# logits, ids_of_candidates_from_batch = self.decoder(embedded_data_points, mentions)
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence

# my_config = GPT2Config(n_positions=30, # maximum sequence length
#                                n_embd= 768, # dim of embeddings and hidden states (2* encoder dim if I do pooling, otherwise same as encoder dim)
#                                n_layer= 4, # number of hidden layers
#                                )
# transformer_decoder = GPT2Model(my_config)
#
# # num_examples, sequence_length, vector dim
# input_vector = torch.randn(768)
# input_vector.unsqueeze_(0)
# input_vector.unsqueeze_(0)
#
# output = transformer_decoder.forward(inputs_embeds=input_vector)
#
# print(output)
# print(output.size())

class GenerativeDecoder(torch.nn.Module):

    def __init__(self, list_of_labels):
        my_config = GPT2Config(n_positions=40, # maximum sequence length
                               n_embd= 2*768, # dim of embeddings and hidden states (2* encoder dim if I do pooling, otherwise same as encoder dim)
                               n_layer= 4, # number of hidden layers
                               # embd_pdrop=0, # TODO: this is only for testing, later I should leave it at the default value 0.1
                               # resid_pdrop=0,
                               # attn_pdrop=0,
                               # summary_first_dropout=0
                               )
        super().__init__()

        self.transformer_decoder = GPT2Model(my_config)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.final_linear_layer = torch.nn.Linear(my_config.n_embd, my_config.vocab_size , bias=False)

        self.label_to_input_ids = {}
        for label in list_of_labels:
            #label = label.replace('_', ' ')
            input_ids = self.tokenizer(label)['input_ids']
            self.label_to_input_ids[label] = torch.tensor(input_ids + [self.tokenizer.eos_token_id])

        self.softmax = torch.nn.Softmax(dim=1)


    def decode(self, embedded_data_points, mentions, labels):

        # num mentions, 1, mention_emb_dim
        hidden_states_encoder = embedded_data_points.unsqueeze(1)

        # now, for teacher forcing we need to add the gold labels to the input
        labels_input_ids = [self.label_to_input_ids[label[0]] for label in labels]
        label_embedding_vectors = [self.transformer_decoder.wte(ids[:-1]) for ids in labels_input_ids]

        # num mentions, max tokenized label, mention_emb_dim
        padded_label_embedding_vectors = pad_sequence(label_embedding_vectors, batch_first=True)

        attention_mask = torch.tensor([[1] * (vec.size()[0] + 1) + [0]*(padded_label_embedding_vectors.size()[1] - vec.size()[0]) for vec in label_embedding_vectors])

        decoder_input_vectors = torch.cat([hidden_states_encoder, padded_label_embedding_vectors], dim=1)

        output = self.transformer_decoder.forward(inputs_embeds=decoder_input_vectors, attention_mask=attention_mask)

        # now we compute the predictions
        logits = self.final_linear_layer(output.last_hidden_state)

        return logits, attention_mask, labels_input_ids

    def generative_inference(self, embedded_mention, max_num_steps=6, prefix_tree: Dict = None, beam_size=5):

        #past_key_values = None

        with torch.no_grad():
            # first step:
            output = self.transformer_decoder.forward(inputs_embeds=embedded_mention)

            # beams
            logits = self.final_linear_layer(output.last_hidden_state[-1, :])
            logits[self.tokenizer.eos_token_id] = -torch.inf # exclude end_of_sentence token for first prediction

            if prefix_tree:
                sensible_continuations = list(prefix_tree.keys())
                # TODO: if sensible continuations has length 1 I do not need to contunue prediction but can directly recursively scroll through the tree
                masked_logits = -torch.inf * torch.ones(logits.size(), requires_grad=True, device=flair.device)
                masked_logits[sensible_continuations] = logits[sensible_continuations]

                output_probabilities = torch.nn.functional.softmax(masked_logits, dim=0)

                probabilities, indices = torch.topk(output_probabilities, beam_size, dim=0)

                beam_sequences = [[[index.item()], torch.log(prob), prefix_tree[index.item()]] for index, prob in
                                  zip(indices, probabilities)]

                #prefix_tree = prefix_tree[output_id.item()]

            else:
                output_probabilities = torch.nn.functional.softmax(logits, dim=0)

                probabilities, indices = torch.topk(output_probabilities, beam_size, dim=0)

                beam_sequences = [[[index.item()], torch.log(prob)] for index, prob in zip(indices, probabilities)] # TODO: add prefix subtree to each sequence???

            final_candidates = []

            # create next input
            beam_mention_embedding = embedded_mention.repeat(beam_size, 1, 1) # initial embedding repeated beam_size times
            indices_embeddings = self.transformer_decoder.wte(indices).unsqueeze(1)

            embedded_beams = torch.cat((beam_mention_embedding, indices_embeddings), dim=1)

            # next step
            # size: #beams, 2, hidden_dim
            output = self.transformer_decoder.forward(inputs_embeds=embedded_beams)

            for gen_step in range(1, max_num_steps):

                #past_key_values = output.past_key_values

                logits = self.final_linear_layer(output.last_hidden_state[:,-1, :])

                if prefix_tree:
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

                scores, indices = torch.topk(log_probs, beam_size, dim=1) # TODO: decreasing beam_size??!!

                all_expanded_candidates = []
                for i, seq in enumerate(beam_sequences):
                    for index, score in zip(indices[i, :], scores[i, :]):
                        if not score == -torch.inf and not torch.isnan(score):
                            if prefix_tree:
                                candidate = [seq[0] + [index.item()], seq[1] + score, seq[2][index.item()]]
                            else:
                                candidate = [seq[0] + [index.item()], seq[1] + score]
                            all_expanded_candidates.append(candidate)

                ordered_candidates = sorted(all_expanded_candidates, key=lambda tup:tup[1], reverse=True)

                beam_sequences = []
                new_indices = []
                for cand in ordered_candidates:
                    if cand[0][-1] == self.tokenizer.eos_token_id: # end of sequence predicted
                        cand[1] = cand[1]/len(cand[0])
                        final_candidates.append(cand)
                    else:
                        beam_sequences.append(cand)
                        new_indices.append(cand[0])

                beam_sequences = beam_sequences[:beam_size]
                new_indices = new_indices[:beam_size]

                # TODO: when do we stop?
                # I fear that small sequences will be preferred
                # Especially if we use a prefix tree there will be many finished sequences early on
                # But luckily this only concerns the testing stage I will have enough possibilities to test, given a trained model
                # One possibility would be:
                # as soon as the final_list is large enough we stop and finish the beam_size times many unfisnished sequences in the beam greedily
                # i.e. until each single beam produces an eos token
                if len(final_candidates) >= beam_size**2 or not beam_sequences:
                    break

                # do the next decoding pass
                new_embedds = self.transformer_decoder.wte(torch.tensor(new_indices))
                beam_mention_embedding = embedded_mention.repeat(len(new_indices), 1, 1)

                embedded_beams = torch.cat((beam_mention_embedding, new_embedds), dim=1)

                output = self.transformer_decoder.forward(inputs_embeds=embedded_beams)

        # add a sequence without eos-token to make sure at least one sequence is contained in final sequence list
        if beam_sequences:
            cand = beam_sequences[0]
            cand[1] = cand[1]/len(cand[0])
            final_candidates.append(cand)

        # TODO: Here I could add a greedy decoding for the remaining unfinished sequences in the beam

        ordered_final_candidates = sorted(final_candidates, key=lambda tup: tup[1], reverse=True)

        generated_sequence = ordered_final_candidates[0][0]
        text_generated_sequence = [self.tokenizer.convert_ids_to_tokens(int(output_id)) for output_id in generated_sequence]

        # transform ids to text output
        output_string = ''.join(text_generated_sequence)

        return output_string


# TODO: Tree with classes
"""
class BinaryTreeNode:
  def __init__(self, data):
    self.data = data
    self.leftChild = None
    self.rightChild=None
"""

class PrefixTree(object):
    def __init__(self, sequences: List[List[int]]):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                PrefixTree._add_to_trie(sequence, self.trie_dict)
                self.len += 1

    def add(self, sequence: List[int]):
        PrefixTree._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict,
        )

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            PrefixTree._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
            prefix_sequence: List[int],
            trie_dict: Dict,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
            )
        else:
            return []

    def __len__(self):
        return self.len


# list_
# prefix_tree = PrefixTree()

# from flair.embeddings import TransformerWordEmbeddings
# from flair.datasets import NEL_ENGLISH_AIDA
# import flair
# flair.device = 'cpu'
# import random
# random.seed(11)
#
#
# emb = TransformerWordEmbeddings(
#                 model='distilbert-base-uncased',
#                 layers="-1",
#                 subtoken_pooling="first",
#                 fine_tune=True,
#                 use_context=False,
#             )
#
# from flair.datasets import ColumnCorpus
#
# corp = ColumnCorpus(data_folder='C:\\Users\\Marcel\\Desktop\\tmp_arbeit\\train_generative_model\\',
#               column_format={0: "text", 2: "nel"},
#               column_delimiter='\t',
#               train_file='t.conll',
#               sample_missing_splits=False,
#               in_memory=True
#               )
# label_dictionary = corp.make_label_dictionary('nel')
#
# decoder = GenerativeDecoder(list_of_labels=label_dictionary.get_items())

####
# # candidate lists
# import pickle
# with open('C:\\Users\\Marcel\\Desktop\\Arbeit\\Task\\Entitiy_Linking\\my_dataset_repo\\ED_Dataset\\train_data\\zelda_mention_entities_counter.pickle', 'rb') as handle:
#     mention_entities_counter = pickle.load(handle)
#
# candidate_generation_dictionary = {}
# for mention in mention_entities_counter:
#     candidate_generation_dictionary[mention] = list(mention_entities_counter[mention].keys())
#
# # # mention to actual mentions
# # with open('/glusterfs/dfs-gfs-dist/milichma/el_test_data/train_data/cg/cg_with_wikidata/merged/mention_to_mentions.pickle', 'rb') as handle:
# #     mention_to_mentions = pickle.load(handle)
#
# candidate_generation_method = CandidateGenerationMethod(candidate_dict=candidate_generation_dictionary)
#
# # create candidate decoder
# decoder = CandidateDecoder(entity_embedding_size=200,
#              mention_embedding_size=2*emb.embedding_length,
#              entity_dictionary= label_dictionary,
#              candidate_generation_method=candidate_generation_method,
#             use_minus_infinity_instead_of_selection=True)
####


# sentences = [corp.train[11]]
#
# linker = EntityLinker(emb,label_dictionary=label_dictionary, label_type='nel', pooling_operation='first_last', decoder=decoder)
#
# embedded_data_points, labels, data_points = linker.forward_pass(sentences, for_prediction=True)
#
#
#
# # push embedded_data_points through decoder to get the scores
# mentions = [span.text for span in data_points]
#
# #linker.decoder.forward(embedded_data_points, mentions)
# embedded_mentions, span_labels, data_points = linker.forward_pass(sentences, for_prediction=True)
#
# #logits, ids_of_candidates_from_batch = linker.decoder.decode(embedded_mentions, mentions)
# #output = linker.decoder.decode(embedded_mentions, mentions, labels)
#
# #loss = linker.forward_loss(sentences=sentences)
#
# #mention_embedding= embedded_mentions[0,:].unsqueeze(0)
# #linker.decoder.generative_inference(embedded_mention=mention_embedding, max_num_steps=5 )
#
# prefix_tree = PrefixTree(linker.decoder.label_to_input_ids[title].tolist() for title in label_dictionary.get_items()).trie_dict
#
# print(linker.predict(sentences, prefix_tree=prefix_tree))
# print(sentences[0])
# #print(loss)

