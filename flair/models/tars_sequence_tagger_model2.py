import logging

from pathlib import Path
from typing import List, Union, Optional, Dict, Set

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import math
import random

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
import flair.nn
from flair.data import Dictionary, Sentence, Label
from flair.datasets import SentenceDataset, DataLoader
from flair.training_utils import Metric, Result, store_embeddings

log = logging.getLogger("flair")

class TARSSequenceTagger2(flair.nn.Model):
    # tags used in BIO2 format: B, I, O
    static_tag_beginning = "B"
    static_tag_inside = "I"
    static_tag_outside = "O"

    static_tag_type = "tars_tag"
    static_adhoc_task_identifier = "ADHOC_TASK_DUMMY"

    def __init__(
            self,
            tag_dictionary: Dictionary,
            tag_type: str,
            task_name: str,
            dropout: float = 0.0,
            word_dropout: float = 0.05,
            locked_dropout: float = 0.5,
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
            num_negative_tags_to_sample: int = 2,
            batch_size: int = 16,
            transformer_word_embeddings: str = 'bert-base-uncased'
    ):
        """
        Initializes a TARSSequenceTagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param locked_dropout: locked dropout probability
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for classes (tags) for the loss function
        (if any tag's weight is unspecified it will default to 1.0)
        :num_negative_tags_to_sample: number of negative tags to sample for each
        positive tags against a sentence during training. Defaults to 2 negative
        tags for each positive tag. The model would sample all the negative tags
        if None is passed. That slows down the training considerably.
        :param task_name: a string depicting the name of the task
        :param batch_size: batch size for forward pass while using BERT
        :param transformer_word_embeddings: name of the pre-trained transformer model e.g.,
        'bert-base-uncased' etc
        """

        super(TARSSequenceTagger2, self).__init__()

        # prepare transformer word embeddings
        from flair.embeddings.token import TransformerWordEmbeddings
        self.transformer_word_embeddings_type: str = transformer_word_embeddings
        self.transformer_word_embeddings = TransformerWordEmbeddings(
            model=transformer_word_embeddings,
            fine_tune=True,
        )

        # all stats are required for state dict
        self.batch_size = batch_size

        # prepare tars tag dictionary
        self.tars_tag_dictionary = Dictionary(add_unk=False)
        self.tars_tag_dictionary.add_item(self.static_tag_beginning) # index 0
        self.tars_tag_dictionary.add_item(self.static_tag_inside) # index 1
        self.tars_tag_dictionary.add_item(self.static_tag_outside)  # index 2

        # tag specific variables
        self.num_negative_tags_to_sample = num_negative_tags_to_sample
        self.tag_nearest_map = None
        self.cleaned_up_tags = {}

        # Store task specific tags since TARS can handle multiple tasks
        self.current_task = None
        self.task_specific_attributes = {}
        self.add_and_switch_to_new_task(task_name, tag_dictionary, tag_type, beta, loss_weights)

        # linear layer
        self.linear = torch.nn.Linear(self.transformer_word_embeddings.embedding_length, len(self.tars_tag_dictionary))

        # dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
        if word_dropout > 0.0:
            self.word_dropout_three_dims = flair.nn.WordDropout(word_dropout, dims=3)
            self.word_dropout_four_dims = flair.nn.WordDropout(word_dropout, dims=4)
        if locked_dropout > 0.0:
            self.locked_dropout_three_dims = flair.nn.LockedDropout(locked_dropout, dims=3)
            self.locked_dropout_four_dims = flair.nn.LockedDropout(locked_dropout, dims=4)

        # F-beta score
        self.beta = beta

        # loss weights
        self.weight_dict = loss_weights
        if loss_weights is not None:
            n_classes = len(self.tag_dictionary)
            weight_list = [1. for i in range(n_classes)]
            for i, tag in enumerate(self.tag_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "task_name": self.current_task,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
            "num_negative_tags_to_sample": self.num_negative_tags_to_sample,
            "batch_size": self.batch_size,
            "transformer_word_embeddings": self.transformer_word_embeddings_type,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if "use_word_dropout" not in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if "use_locked_dropout" not in state.keys()
            else state["use_locked_dropout"]
        )
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]

        model = TARSSequenceTagger2(
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            task_name=state["task_name"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            beta=beta,
            loss_weights=weights,
            num_negative_tags_to_sample=state["num_negative_tags_to_sample"],
            batch_size=state["batch_size"],
            transformer_word_embeddings=state["transformer_word_embeddings"],
        )
        model.load_state_dict(state["state_dict"])
        return model

    def add_and_switch_to_new_task(self,
                                   task_name,
                                   tag_dictionary: Union[List, Set, Dictionary],
                                   tag_type: str = None,
                                   beta: float = 1.0,
                                   weight_dict: Dict[str, float] = None,
                                   ):
        """
        Adds a new task to an existing TARS model. Sets necessary attributes and finally 'switches'
        to the new task. Parameters are similar to the constructor except for model choice, batch
        size and negative sampling. This method does not store the resultant model onto disk.
        :param task_name: a string depicting the name of the task
        :param tag_dictionary: dictionary of the tags you want to predict
        """
        if task_name in self.task_specific_attributes:
            log.warning("Task `%s` already exists in TARS model. Switching to it.", task_name)
        else:
            if isinstance(tag_dictionary, (list, set)):
                tag_dictionary = TARSSequenceTagger2._make_ad_hoc_tag_dictionary(tag_dictionary)
                # insert tags missing in tag_dictionary (e.g. if only I-PER or only B-PER is in it, but the other not)
                for tag in tag_dictionary.idx2item:
                    tag = tag.decode("utf-8")
                    tag_prefix, tag_no_prefix = self._split_tag(tag)
                    if tag_prefix == "B":
                        tag_dictionary.add_item("I-" + tag_no_prefix)
                    elif tag_prefix == "I":
                        tag_dictionary.add_item("B-" + tag_no_prefix)
            self.task_specific_attributes[task_name] = {}
            self.task_specific_attributes[task_name]['tag_dictionary'] = tag_dictionary
            self.task_specific_attributes[task_name]['tag_type'] = tag_type
            self.task_specific_attributes[task_name]['beta'] = beta
            # TODO: further implement here: loss-weights? dropout? num_negative_samples? (all of them required for further training)
            self.task_specific_attributes[task_name]['weight_dict'] = weight_dict
        self.switch_to_task(task_name)

    def switch_to_task(self, task_name):
        """
        Switches to a task which was previously added.
        """
        if task_name not in self.task_specific_attributes:
            log.error("Provided `%s` does not exist in the model. Consider calling "
                      "`add_and_switch_to_new_task` first.", task_name)
        else:
            self.current_task = task_name
            self.tag_dictionary = self.task_specific_attributes[task_name]['tag_dictionary']
            self.tag_dictionary_no_prefix = self._get_tag_dictionary_no_prefix()
            self.tag_type = self.task_specific_attributes[task_name]['tag_type']
            self.beta = self.task_specific_attributes[task_name]['beta']
            # TODO: further implement here: loss-weights? dropout? num_negative_samples?
            self.weight_dict = self.task_specific_attributes[task_name]['weight_dict']
            if self.weight_dict is not None:
                n_classes = len(self.tag_dictionary)
                weight_list = [1. for i in range(n_classes)]
                for i, tag in enumerate(self.tag_dictionary.get_items()):
                    if tag in self.weight_dict.keys():
                        weight_list[i] = self.weight_dict[tag]
                self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
            else:
                self.loss_weights = None

    def _drop_task(self, task_name):
        if task_name in self.task_specific_attributes:
            if self.current_task == task_name:
                log.error("`%s` is the current task."
                          " Switch to some other task before dropping this.", task_name)
            else:
                self.task_specific_attributes.pop(task_name)
        else:
            log.warning("No task exists with the name `%s`.", task_name)

    def list_existing_tasks(self):
        """
        Lists existing tasks in the loaded TARS model on the console.
        """
        print("Existing tasks are:")
        for task_name in self.task_specific_attributes:
            print(task_name)

    def _get_cleaned_up_tag(self, tag):
        """
        Does some basic clean up of the provided tags, stores them, looks them up.
        """
        if tag not in self.cleaned_up_tags:
            self.cleaned_up_tags[tag] = tag.replace("_", " ")
        return self.cleaned_up_tags[tag]

    def _split_tag(self, tag: str):
        if tag == "O":
            return tag, None
        elif "-" in tag:
            tag_split = tag.split("-")
            return tag_split[0], "-".join(tag_split[1:])
        else:
            return None, None

    def _get_tag_dictionary_no_prefix(self):
        candidate_tag_list = []
        for tag in self.tag_dictionary.idx2item:
            tag = tag.decode("utf-8")
            spl = tag.split("-")
            if len(spl) == 1: # no tag was found
                print("no - was found in a tag:")
                print(tag)
                print()
            elif tag != "O": # tag found that is either B or I
                candidate_tag_list.append("-".join(spl[1:]))
        candidate_tag_list = self._remove_not_unique_items_from_list(candidate_tag_list)

        tag_dictionary_no_prefix: Dictionary = Dictionary(add_unk=False)
        for tag in candidate_tag_list:
            tag_dictionary_no_prefix.add_item(tag)

        return tag_dictionary_no_prefix

    def _compute_tag_similarity_for_current_epoch(self):
        """
        Compute the similarity between all tags for better sampling of negatives
        """

        # get and embed all tags by making a Sentence object that contains only the label text
        all_tags = [tag.decode("utf-8") for tag in self.tag_dictionary_no_prefix.idx2item]
        tag_sentences = [Sentence(self._get_cleaned_up_tag(tag)) for tag in all_tags]

        self.transformer_word_embeddings.embed(tag_sentences)
        encodings_np = []
        for sentence in tag_sentences:
            embed_sum = np.zeros(self.transformer_word_embeddings.embedding_length)
            for token in sentence:
                embed = token.get_embedding().cpu().detach().numpy()
                embed_sum += embed
            embed_mean = embed_sum / len(sentence)
            encodings_np.append(embed_mean)
        normalized_encoding = minmax_scale(encodings_np)

        # compute similarity matrix
        similarity_matrix = cosine_similarity(normalized_encoding)

        # the higher the similarity, the greater the chance that a label is
        # sampled as negative example
        negative_tag_probabilities = {}
        for row_index, tag in enumerate(all_tags):
            negative_tag_probabilities[tag] = {}
            for column_index, other_tag in enumerate(all_tags):
                if tag != other_tag:
                    negative_tag_probabilities[tag][other_tag] = \
                        similarity_matrix[row_index][column_index]
        self.tag_nearest_map = negative_tag_probabilities

    def train(self, mode=True):
        if mode and self.num_negative_tags_to_sample is not None:
            self._compute_tag_similarity_for_current_epoch()
        super(TARSSequenceTagger2, self).train(mode)

    def _get_random_tags(self):
        random_samples_decoded = []
        sample_num = min(self.num_negative_tags_to_sample, len(self.tag_dictionary_no_prefix.idx2item))
        rand_samples = random.sample(self.tag_dictionary_no_prefix.idx2item, sample_num)
        for item in rand_samples:
            random_samples_decoded.append(item.decode("UTF-8"))
        return random_samples_decoded

    def _get_nearest_tags_for(self, tags):
        nearest_tags = []

        if len(tags) == 0:
            return self._get_random_tags()

        for tag in tags:
            plausible_tags = []
            plausible_tag_probabilities = []
            for plausible_tag in self.tag_nearest_map[tag]:
                if plausible_tag in nearest_tags or plausible_tag in tags:
                    continue
                else:
                    plausible_tags.append(plausible_tag)
                    plausible_tag_probabilities.append( \
                        self.tag_nearest_map[tag][plausible_tag])

            # make sure the probabilities always sum up to 1
            plausible_tag_probabilities = np.array(plausible_tag_probabilities, dtype='float64')
            plausible_tag_probabilities += 1e-08
            plausible_tag_probabilities /= np.sum(plausible_tag_probabilities)

            if len(plausible_tags) > 0:
                num_samples = min(self.num_negative_tags_to_sample, len(plausible_tags))
                sampled_negative_tags = np.random.choice(plausible_tags,
                                                         num_samples,
                                                         replace=False,
                                                         p=plausible_tag_probabilities)
                nearest_tags.extend(sampled_negative_tags)

        return nearest_tags

    def _get_tars_formatted_sentence(self, tag: str, sentence: Sentence):
        tag_text_pair = " ".join([self._get_cleaned_up_tag(tag),
                                  self.transformer_word_embeddings.tokenizer.sep_token,
                                  sentence.to_tokenized_string()])
        tag_text_pair_sentence = Sentence(tag_text_pair, use_tokenizer=False)

        offset = len(tag_text_pair_sentence) - len(sentence) # amount of tokens of tag and sep_token
        for idx_in_new_sent in range(offset, len(tag_text_pair_sentence)): # ignore tokens of tag and sep_token
            idx_in_old_sent = idx_in_new_sent - offset
            old_tag = sentence[idx_in_old_sent].get_tag(self.tag_type).value
            old_tag_prefix, old_tag_no_prefix = self._split_tag(old_tag)
            if old_tag_prefix != None:  # else the word in sentence is untagged at prediction moment
                if old_tag_no_prefix == tag and old_tag_prefix == self.static_tag_beginning:
                    tag_text_pair_sentence[idx_in_new_sent].add_tag(self.static_tag_type, self.static_tag_beginning)
                elif old_tag_no_prefix == tag and old_tag_prefix == self.static_tag_inside:
                    tag_text_pair_sentence[idx_in_new_sent].add_tag(self.static_tag_type, self.static_tag_inside)
                else:
                    tag_text_pair_sentence[idx_in_new_sent].add_tag(self.static_tag_type, self.static_tag_outside)

        return tag_text_pair_sentence

    def _get_tars_formatted_sentences(self, sentences, full_forward=False):
        tag_text_pairs = []

        for sentence in sentences:
            tag_text_pairs_for_sentence = []

            if not full_forward and self.num_negative_tags_to_sample is not None:
                tags_of_sentence = [self._split_tag(token.get_tag(self.tag_type).value)[1] for token in sentence]
                tags_of_sentence = [tag for tag in tags_of_sentence if tag is not None]
                tags_of_sentence = self._remove_not_unique_items_from_list(tags_of_sentence)
                sampled_tags_not_in_sentence = self._get_nearest_tags_for(tags_of_sentence)
                sampled_tags_not_in_sentence = self._remove_not_unique_items_from_list(sampled_tags_not_in_sentence)
                for tag in tags_of_sentence:
                    tag_text_pairs_for_sentence.append( \
                        self._get_tars_formatted_sentence(tag, sentence))
                for tag in sampled_tags_not_in_sentence:
                    tag_text_pairs_for_sentence.append( \
                        self._get_tars_formatted_sentence(tag, sentence))
            else:
                all_tags = [tag.decode("utf-8") for tag in self.tag_dictionary_no_prefix.idx2item]
                for tag in all_tags:
                    tag_text_pairs_for_sentence.append( \
                        self._get_tars_formatted_sentence(tag, sentence))
            tag_text_pairs.extend(tag_text_pairs_for_sentence)
        return tag_text_pairs

    def _remove_not_unique_items_from_list(self, l: list):
        new_list = []
        for item in l:
            if item not in new_list:
                new_list.append(item)
        return new_list

    @staticmethod
    def _make_ad_hoc_tag_dictionary(candidate_tag_set: set = None) -> Dictionary:
        """
        Creates a dictionary given a set of candidate tags
        :return: dictionary of tags
        """
        tag_dictionary: Dictionary = Dictionary(add_unk=False)

        if not isinstance(candidate_tag_set, set):
            candidate_tag_set = set(candidate_tag_set)

        for tag in candidate_tag_set:
            tag_dictionary.add_item(tag)

        return tag_dictionary

    def predict_zero_shot(self, sentences, candidate_tag_set):
        """
        Method to make zero shot predictions from the TARS model
        :param sentences: input sentence objects to classify
        :param candidate_tag_set: set of candidate tags
        Defaults to False
        """

        # check if candidate_tag_set is empty
        if candidate_tag_set is None or len(candidate_tag_set) == 0:
            log.warning("Provided candidate_tag_set is empty")
            return

        tag_dictionary = TARSSequenceTagger2._make_ad_hoc_tag_dictionary(candidate_tag_set)

        # note current task
        existing_current_task = self.current_task

        # create a temporary task
        self.add_and_switch_to_new_task(TARSSequenceTagger2.static_adhoc_task_identifier, tag_dictionary, "ner") #TODO: no tag_type needed here?

        try:
            # make zero shot predictions
            self.predict(sentences)
        except:
            log.error("Something went wrong during prediction. Ensure you pass Sentence objects.")
        finally:
            # switch to the pre-existing task
            self.switch_to_task(existing_current_task)
            self._drop_task(TARSSequenceTagger2.static_adhoc_task_identifier)

        return

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            mini_batch_size=32,
            all_tag_prob: bool = False,
            verbose: bool = False,
            tag_name: Optional[str] = None,
            return_loss=False,
            embedding_storage_mode="none",
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
        :param tag_name: set this to change the name of the tag type that is predicted
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if tag_name == None:
            tag_name = self.tag_type

        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, Sentence):
                sentences = [sentences]

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )

            reordered_sentences: List[Union[Sentence, str]] = [
                sentences[index] for index in rev_order_len_index
            ]

            dataloader = DataLoader(
                dataset=SentenceDataset(reordered_sentences), batch_size=mini_batch_size
            )

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            overall_loss = 0
            batch_no = 0
            for batch in dataloader:

                batch_no += 1

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {batch_no}")

                batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                tag_scores, formatted_sentences, sentence_offsets, sentence_rest_lengths = self._forward_four_dims(batch)
                feature = self._transform_tars_scores(tag_scores)
                if return_loss:
                    overall_loss += self._calculate_loss_four_dims(tag_scores, formatted_sentences, sentence_offsets, sentence_rest_lengths)

                tags, all_tags = self._obtain_labels(
                    feature=feature,
                    batch_sentences=batch,
                    get_all_tags=all_tag_prob,
                )

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token.add_tag_label(tag_name, tag)

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (sentence, sent_all_tags) in zip(batch, all_tags):
                    for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                        token.add_tags_proba_dist(tag_name, token_all_tags)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss / batch_no

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences

    def _transform_tars_scores(self, tars_scores):
        # M: num_classes in task (all), N: num_samples (batch_size), L: max_sequence_length, m: num_classes (no prefix)
        # reshape from NxLxmx3 to NxLxM (m is approx. M/2):
        tars_scores = torch.nn.functional.softmax(tars_scores, dim=3)  # softmax onto B-I-O-mapping
        transformed_scores = []
        for i in range(len(tars_scores)):  # N
            transformed_scores.append([])
            for j in range(len(tars_scores[i])):  # L
                transformed_scores[i].append([])
                tmp = []  # should get the length M (first index represents 0, then following other)
                for tag_idx, tag in enumerate(self.tag_dictionary.idx2item):
                    tmp.append([])
                    tag = tag.decode("utf-8")
                    tag_prefix, tag_no_prefix = self._split_tag(tag)
                    if tag_prefix == "O":
                        """
                        find highest B, or I value for the word and use the O value next to it.
                        """
                        max_k_idx = 0
                        max = -math.inf
                        for k in range(len(tars_scores[i][j])):
                            for l in range(len(tars_scores[i][j][k])):
                                if l != 2 and max < tars_scores[i][j][k][l]:  # index 2 represents "O"
                                    max = tars_scores[i][j][k][l]
                                    max_k_idx = k
                        tmp[tag_idx] = tars_scores[i][j][max_k_idx][2]  # set it to the value next to the hightest I, or B value
                    else:
                        tag_no_prefix_idx = self.tag_dictionary_no_prefix.item2idx[tag_no_prefix.encode("UTF-8")]
                        tag_prefix_idx = self.tars_tag_dictionary.item2idx[tag_prefix.encode("UTF-8")]
                        tmp[tag_idx] = tars_scores[i][j][tag_no_prefix_idx][tag_prefix_idx]
                transformed_scores[i][j] = tmp
        transformed_scores = torch.tensor(transformed_scores, dtype=torch.float, device=flair.device)
        return transformed_scores

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence], sort=True
    ) -> torch.tensor:
        if isinstance(data_points, Sentence):
            data_points = [data_points]
        tag_scores, formatted_sentences, sentence_offsets, sentence_rest_lengths = self._forward_three_dims(data_points)

        return self._calculate_loss_three_dims(tag_scores, formatted_sentences, sentence_offsets, sentence_rest_lengths)

    def forward(self, sentences: List[Sentence]):
        tag_scores, _, _, _ = self._forward_four_dims(sentences)
        transformed_scores = self._transform_tars_scores(tag_scores)
        return transformed_scores

    # N x L x M x 2
    def _forward_four_dims(self, sentences: List[Sentence]):
        # Transform input data into TARS format
        formatted_sentences = self._get_tars_formatted_sentences(sentences, full_forward=True)

        sentence_offsets = []
        sentence_rest_lengths = []
        for sent in formatted_sentences:
            sep_token_reached = False
            offset = 0
            rest_length = 0
            for tkn in sent:
                if not sep_token_reached:
                    offset += 1
                    if tkn.text == self.transformer_word_embeddings.tokenizer.sep_token:
                        sep_token_reached = True
                else:
                    rest_length += 1
            sentence_offsets.append(offset)
            sentence_rest_lengths.append(rest_length)
        longest_token_sequence_in_batch = max(sentence_rest_lengths)

        m = len(self.tag_dictionary_no_prefix.item2idx)
        pre_allocated_zero_tensor = torch.zeros(
            self.transformer_word_embeddings.embedding_length * longest_token_sequence_in_batch * m,  # E * L * M
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        self.transformer_word_embeddings.embed(formatted_sentences)
        for sentence_in_batch in range(len(sentences)):  # for each in 0,...,N batch sentences
            for token_in_sentence in range(longest_token_sequence_in_batch):  # for each in the 0,...,L (or less) tokens
                for tag_idx in range(m):  # for each in the 0,...,M tags
                    sent_idx = sentence_in_batch * m + tag_idx  # index of the current sentence in batch with prepended current tag
                    sent = formatted_sentences[sent_idx]

                    if token_in_sentence < sentence_rest_lengths[sent_idx]:  # token existiert in diesem satz
                        tkn_idx = token_in_sentence + sentence_offsets[sent_idx]  # wo liegt der fuer dieses label?
                        tkn = sent[tkn_idx]
                        embed = tkn.get_embedding()
                        all_embs.append(embed)
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentences[sentence_in_batch])
            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    :self.transformer_word_embeddings.embedding_length * nb_padding_tokens * m]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),  # N (batch_size)
                longest_token_sequence_in_batch,  # L
                m,  # M
                self.transformer_word_embeddings.embedding_length,
            ]
        )

        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout_four_dims(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout_four_dims(sentence_tensor)

        tag_scores = self.linear(sentence_tensor)
        return tag_scores, formatted_sentences, sentence_offsets, sentence_rest_lengths

    # N+ x L x 2
    def _forward_three_dims(self, sentences: List[Sentence]):
        formatted_sentences = self._get_tars_formatted_sentences(sentences)

        sentence_offsets = []
        sentence_rest_lengths = []
        for sent in formatted_sentences:
            sep_token_reached = False
            offset = 0
            rest_length = 0
            for tkn in sent:
                if not sep_token_reached:
                    offset += 1
                    if tkn.text == self.transformer_word_embeddings.tokenizer.sep_token:
                        sep_token_reached = True
                else:
                    rest_length += 1
            sentence_offsets.append(offset)
            sentence_rest_lengths.append(rest_length)
        longest_token_sequence_in_batch = max(sentence_rest_lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.transformer_word_embeddings.embedding_length * longest_token_sequence_in_batch,  # E * L
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        self.transformer_word_embeddings.embed(formatted_sentences)
        for formatted_sentence_idx, formatted_sentence in enumerate(
                formatted_sentences):  # for each in 0,...,N+ formatted sentences
            for token_in_sentence in range(longest_token_sequence_in_batch):  # for each in the 0,...,L (or less) tokens
                if token_in_sentence < sentence_rest_lengths[formatted_sentence_idx]:  # token existiert in diesem satz
                    tkn_idx = token_in_sentence + sentence_offsets[formatted_sentence_idx]
                    tkn = formatted_sentence[tkn_idx]
                    embed = tkn.get_embedding()
                    all_embs.append(embed)
            nb_padding_tokens = longest_token_sequence_in_batch - sentence_rest_lengths[formatted_sentence_idx]
            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    :self.transformer_word_embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(formatted_sentences),  # N+, N <= N+ <= N*M (formatted batch_size)
                longest_token_sequence_in_batch,  # L
                self.transformer_word_embeddings.embedding_length,
            ]
        )

        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout_three_dims(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout_three_dims(sentence_tensor)
        features = self.linear(sentence_tensor)

        return features, formatted_sentences, sentence_offsets, sentence_rest_lengths

    # three dims dual_space loss
    def _calculate_loss_three_dims(
            self, features: torch.tensor, formatted_sentences, sentence_offsets, sentence_rest_lengths
    ) -> float:
        target_tag_list_bio_space: List = []
        for formatted_sentence_idx, formatted_sentence in enumerate(formatted_sentences):
            target_tag_idx_bio_space: List[int] = [
                self.tars_tag_dictionary.get_idx_for_item(
                    formatted_sentence[sentence_offsets[formatted_sentence_idx] + i].get_tag(
                        self.static_tag_type).value)
                for i in range(sentence_rest_lengths[formatted_sentence_idx])
            ]
            target_tag_idx_bio_space_tensor = torch.tensor(target_tag_idx_bio_space, device=flair.device)
            target_tag_list_bio_space.append(target_tag_idx_bio_space_tensor)
        score = 0
        for sentence_feats, sentence_tags, sentence_length in zip(
                features, target_tag_list_bio_space, sentence_rest_lengths
        ):
            sentence_feats = sentence_feats[:sentence_length]
            score += torch.nn.functional.cross_entropy(
                sentence_feats, sentence_tags
            )
        score /= len(features)
        return score

    # four dims dual_space loss
    def _calculate_loss_four_dims(
            self, features: torch.tensor, formatted_sentences, sentence_offsets, sentence_rest_lengths
    ) -> float:
        n = len(features)
        m = len(self.tag_dictionary_no_prefix.item2idx)
        target_tag_list_bio_space: List = []
        for sentence_in_batch in range(n):  # for each in 0,...,N batch sentences
            for tag_idx in range(m):  # for each in the 0,...,M tags
                formatted_sentence_idx = sentence_in_batch * m + tag_idx
                formatted_sentence = formatted_sentences[formatted_sentence_idx]
                target_tag_idx_bio_space: List[int] = [
                    self.tars_tag_dictionary.get_idx_for_item(
                        formatted_sentence[sentence_offsets[formatted_sentence_idx] + i].get_tag(
                            self.static_tag_type).value)
                    for i in range(sentence_rest_lengths[formatted_sentence_idx])
                ]
                target_tag_idx_bio_space_tensor = torch.tensor(target_tag_idx_bio_space, device=flair.device)
                target_tag_list_bio_space.append(target_tag_idx_bio_space_tensor)

        score = 0
        for sentence_in_batch in range(n):  # for each in 0,...,N batch sentences
            for tag_idx in range(m):  # for each in the 0,...,M tags
                formatted_sentence_idx = sentence_in_batch * m + tag_idx
                sentence_length = sentence_rest_lengths[formatted_sentence_idx]
                sentence_feats = features[sentence_in_batch][:sentence_length, tag_idx]
                sentence_tags = target_tag_list_bio_space[formatted_sentence_idx]
                score += torch.nn.functional.cross_entropy(
                    sentence_feats, sentence_tags
                )
        score /= n * m
        return score

    def _obtain_labels(
            self,
            feature: torch.Tensor,
            batch_sentences: List[Sentence],
            get_all_tags: bool,
    ) -> (List[List[Label]], List[List[List[Label]]]):
        """
        Returns a tuple of two lists:
         - The first list corresponds to the most likely `Label` per token in each sentence.
         - The second list contains a probability distribution over all `Labels` for each token
           in a sentence for all sentences.
        """

        lengths: List[int] = [len(sentence.tokens) for sentence in batch_sentences]

        tags = []
        all_tags = []
        feature = feature.cpu()
        for index, length in enumerate(lengths):
            feature[index, length:] = 0
        softmax_batch = F.softmax(feature, dim=2).cpu()
        scores_batch, prediction_batch = torch.max(softmax_batch, dim=2)
        feature = zip(softmax_batch, scores_batch, prediction_batch)

        for feats, length in zip(feature, lengths):
            softmax, score, prediction = feats
            confidences = score[:length].tolist()
            tag_seq = prediction[:length].tolist()
            scores = softmax[:length].tolist()

            tags.append(
                [
                    Label(self.tag_dictionary.get_item_for_index(tag), conf)
                    for conf, tag in zip(confidences, tag_seq)
                ]
            )

            if get_all_tags:
                all_tags.append(
                    [
                        [
                            Label(
                                self.tag_dictionary.get_item_for_index(score_id), score
                            )
                            for score_id, score in enumerate(score_dist)
                        ]
                        for score_dist in scores
                    ]
                )

        return tags, all_tags

    def __str__(self): # TODO
        return super(flair.nn.Model, self).__str__().rstrip(')') + \
               f'  (beta): {self.beta}\n' + \
               f'  (weights): {self.weight_dict}\n' + \
               f'  (weight_tensor) {self.loss_weights}\n)'

    def _requires_span_F1_evaluation(self) -> bool:
        span_F1 = False
        for item in self.tag_dictionary.get_items():
            if item.startswith('B-'):
                span_F1 = True
        return span_F1

    # TODO: methods that have to be overseen and maybe reworked:

    def _evaluate_with_span_F1(self, data_loader, embedding_storage_mode, mini_batch_size, out_path):
        eval_loss = 0

        batch_no: int = 0

        metric = Metric("Evaluation", beta=self.beta)

        lines: List[str] = []

        y_true = []
        y_pred = []

        for batch in data_loader:

            # predict for batch
            loss = self.predict(batch,
                                embedding_storage_mode=embedding_storage_mode,
                                mini_batch_size=mini_batch_size,
                                tag_name='predicted',
                                return_loss=True)
            eval_loss += loss
            batch_no += 1

            for sentence in batch:

                # make list of gold tags
                gold_spans = sentence.get_spans(self.tag_type)
                gold_tags = [(span.tag, repr(span)) for span in gold_spans]

                # make list of predicted tags
                predicted_spans = sentence.get_spans("predicted")
                predicted_tags = [(span.tag, repr(span)) for span in predicted_spans]

                # check for true positives, false positives and false negatives
                for tag, prediction in predicted_tags:
                    if (tag, prediction) in gold_tags:
                        metric.add_tp(tag)
                    else:
                        metric.add_fp(tag)

                for tag, gold in gold_tags:
                    if (tag, gold) not in predicted_tags:
                        metric.add_fn(tag)

                tags_gold = []
                tags_pred = []

                # also write to file in BIO format to use old conlleval script
                if out_path:
                    for token in sentence:
                        # check if in gold spans
                        gold_tag = 'O'
                        for span in gold_spans:
                            if token in span:
                                gold_tag = 'B-' + span.tag if token == span[0] else 'I-' + span.tag
                        tags_gold.append(gold_tag)

                        predicted_tag = 'O'
                        # check if in predicted spans
                        for span in predicted_spans:
                            if token in span:
                                predicted_tag = 'B-' + span.tag if token == span[0] else 'I-' + span.tag
                        tags_pred.append(predicted_tag)

                        lines.append(f'{token.text} {gold_tag} {predicted_tag}\n')
                    lines.append('\n')

                y_true.append(tags_gold)
                y_pred.append(tags_pred)

        if out_path:
            with open(Path(out_path), "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        eval_loss /= batch_no

        detailed_result = (
            "\nResults:"
            f"\n- F1-score (micro) {metric.micro_avg_f_score():.4f}"
            f"\n- F1-score (macro) {metric.macro_avg_f_score():.4f}"
            '\n\nBy class:'
        )

        for class_name in metric.get_classes():
            detailed_result += (
                f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                f"fn: {metric.get_fn(class_name)} - precision: "
                f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                f"f1-score: "
                f"{metric.f_score(class_name):.4f}"
            )

        result = Result(
            main_score=metric.micro_avg_f_score(),
            log_line=f"{metric.precision():.4f}\t{metric.recall():.4f}\t{metric.micro_avg_f_score():.4f}",
            log_header="PRECISION\tRECALL\tF1",
            detailed_results=detailed_result,
        )

        return result, eval_loss

    def evaluate(
            self,
            sentences: Union[List[Sentence], Dataset],
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
    ) -> (Result, float):

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        # if span F1 needs to be used, use separate eval method
        if self._requires_span_F1_evaluation():
            return self._evaluate_with_span_F1(data_loader, embedding_storage_mode, mini_batch_size, out_path)

        # else, use scikit-learn to evaluate
        y_true = []
        y_pred = []
        labels = Dictionary(add_unk=False)

        eval_loss = 0
        batch_no: int = 0

        lines: List[str] = []

        for batch in data_loader:

            # predict for batch
            loss = self.predict(batch,
                                embedding_storage_mode=embedding_storage_mode,
                                mini_batch_size=mini_batch_size,
                                tag_name='predicted',
                                return_loss=True)
            eval_loss += loss
            batch_no += 1

            for sentence in batch:

                for token in sentence:
                    # add gold tag
                    gold_tag = token.get_tag(self.tag_type).value
                    y_true.append(labels.add_item(gold_tag))

                    # add predicted tag
                    predicted_tag = token.get_tag('predicted').value
                    y_pred.append(labels.add_item(predicted_tag))

                    # for file output
                    lines.append(f'{token.text} {gold_tag} {predicted_tag}\n')

                lines.append('\n')

        if out_path:
            with open(Path(out_path), "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        eval_loss /= batch_no

        # use sklearn
        from sklearn import metrics

        # make "classification report"
        target_names = []
        labels_to_report = []
        all_labels = []
        all_indices = []
        for i in range(len(labels)):
            label = labels.get_item_for_index(i)
            all_labels.append(label)
            all_indices.append(i)
            if label == '_' or label == '': continue
            target_names.append(label)
            labels_to_report.append(i)

        # report over all in case there are no labels
        if not labels_to_report:
            target_names = all_labels
            labels_to_report = all_indices

        classification_report = metrics.classification_report(y_true, y_pred, digits=4, target_names=target_names,
                                                              zero_division=1, labels=labels_to_report)

        # get scores
        micro_f_score = round(
            metrics.fbeta_score(y_true, y_pred, beta=self.beta, average='micro', labels=labels_to_report), 4)
        macro_f_score = round(
            metrics.fbeta_score(y_true, y_pred, beta=self.beta, average='macro', labels=labels_to_report), 4)
        accuracy_score = round(metrics.accuracy_score(y_true, y_pred), 4)

        detailed_result = (
                "\nResults:"
                f"\n- F-score (micro): {micro_f_score}"
                f"\n- F-score (macro): {macro_f_score}"
                f"\n- Accuracy (incl. no class): {accuracy_score}"
                '\n\nBy class:\n' + classification_report
        )

        # line for log file
        log_header = "ACCURACY"
        log_line = f"\t{accuracy_score}"

        result = Result(
            main_score=micro_f_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
        )
        return result, eval_loss
