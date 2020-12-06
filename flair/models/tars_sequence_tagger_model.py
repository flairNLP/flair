import logging

from pathlib import Path
from typing import List, Union, Optional, Dict, Set

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
import flair.nn
from flair.data import Dictionary, Sentence, Label
from flair.datasets import SentenceDataset, DataLoader
from flair.training_utils import Metric, Result, store_embeddings

log = logging.getLogger("flair")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"

class TARSSequenceTagger(flair.nn.Model):
    # tags used in BIO2 format: B, I, O
    # static_tag_beginning = "B"
    # static_tag_inside = "I"
    # static_tag_outside = "O"
    static_tag_yes = "YES"
    static_tag_no = "NO"
    static_tag_type = "tars_tag"
    static_adhoc_task_identifier = "adhoc_dummy"

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
            transformer_word_embeddings: str = 'bert-base-uncased',
    ):
        """
        Initializes a TARSSequenceTagger
        :param embeddings: word embeddings used in tagger
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

        super(TARSSequenceTagger, self).__init__()

        # prepare transformer word embeddings
        from flair.embeddings.token import TransformerWordEmbeddings
        self.transformer_word_embeddings = TransformerWordEmbeddings(
            model=transformer_word_embeddings,
            fine_tune=True,
            batch_size=batch_size,
        )
        from flair.embeddings.document import TransformerDocumentEmbeddings
        self.transformer_document_embeddings = TransformerDocumentEmbeddings(
            model=transformer_word_embeddings, # TODO: same values for both available?
            fine_tune=True,
            batch_size=batch_size,
        )

        # prepare BIO2 tag dictionary
        self.tars_tag_dictionary = Dictionary(add_unk=False)
        # self.tars_tag_dictionary.add_item(self.static_tag_beginning)
        # self.tars_tag_dictionary.add_item(self.static_tag_inside)
        # self.tars_tag_dictionary.add_item(self.static_tag_outside)
        self.tars_tag_dictionary.add_item(self.static_tag_yes)
        self.tars_tag_dictionary.add_item(self.static_tag_no)

        # tag specific variables
        self.num_negative_tags_to_sample = num_negative_tags_to_sample
        self.tag_nearest_map = None
        self.cleaned_up_tags = {}

        # Store task specific tags since TARS can handle multiple tasks
        self.current_task = None
        self.task_specific_attributes = {}
        self.add_and_switch_to_new_task(task_name, tag_dictionary, tag_type, beta, loss_weights)

        # dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary
        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

        # linear layer
        self.linear = torch.nn.Linear(self.transformer_word_embeddings.embedding_length, len(self.tars_tag_dictionary))

        # dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
        if word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        if locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

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
            "task_name": self.task_name,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
            "num_negative_tags_to_sample": self.num_negative_tags_to_sample,
            "batch_size": self.batch_size,
            "transformer_word_embeddings": self.transformer_word_embeddings,
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

        model = TARSSequenceTagger(
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
                tag_dictionary = TARSSequenceTagger._make_ad_hoc_tag_dictionary(tag_dictionary)
            self.task_specific_attributes[task_name] = {}
            self.task_specific_attributes[task_name]['tag_dictionary'] = tag_dictionary
            self.task_specific_attributes[task_name]['tag_type'] = tag_type
            self.task_specific_attributes[task_name]['beta'] = beta
            # TODO: further implement here: loss-weights? dropout? num_negative_samples?
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

    def _compute_tag_similarity_for_current_epoch(self):
        """
        Compute the similarity between all tags for better sampling of negatives
        """

        # get and embed all tags by making a Sentence object that contains only the label text
        all_tags = [tag.decode("utf-8") for tag in self.tag_dictionary.idx2item]
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
        """Populate tag similarity map based on cosine similarity before running epoch

        If the `num_negative_tags_to_sample` is set to an integer value then before starting
        each epoch the model would create a similarity measure between the tag names based
        on cosine distances between their BERT encoded embeddings.
        """
        if mode and self.num_negative_tags_to_sample is not None:
            self._compute_tag_similarity_for_current_epoch()
            super(TARSSequenceTagger, self).train(mode)

        super(TARSSequenceTagger, self).train(mode)

    def _get_nearest_tags_for(self, tags):
        nearest_tags = set()

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
                nearest_tags.update(sampled_negative_tags)

        return nearest_tags

    def _get_tars_formatted_sentence(self, tag, sentence: Sentence):
        tag_text_pair = " ".join([self._get_cleaned_up_tag(tag),
                                  self.transformer_word_embeddings.tokenizer.sep_token,
                                  sentence.to_tokenized_string()])
        tag_text_pair_sentence = Sentence(tag_text_pair, use_tokenizer=False)

        offset = len(tag_text_pair_sentence) - len(sentence) # amount of tokens of tag and sep_token
        for idx_in_new_sent in range(offset, len(tag_text_pair_sentence)): # ignore tokens of tag and sep_token
            idx_in_old_sent = idx_in_new_sent - offset
            old_tag = sentence[idx_in_old_sent].get_tag(self.tag_type).value

            if old_tag == tag:
                tag_text_pair_sentence[idx_in_new_sent].add_tag(self.static_tag_type, self.static_tag_yes)
            else:
                tag_text_pair_sentence[idx_in_new_sent].add_tag(self.static_tag_type, self.static_tag_no)
        return tag_text_pair_sentence

    def _get_tars_formatted_sentences(self, sentences):
        tag_text_pairs = []
        all_tags = [tag.decode("utf-8") for tag in self.tag_dictionary.idx2item]
        for sentence in sentences:
            tag_text_pairs_for_sentence = []
            # if self.training and self.num_negative_tags_to_sample is not None:
            #     tags_of_sentence = {token.get_tag(self.tag_type).value for token in sentence \
            #                         if token.get_tag(self.tag_type).value != "O"}
            #     sampled_tags_not_in_sentence = self._get_nearest_tags_for(tags_of_sentence)
            #     for tag in tags_of_sentence:
            #         tag_text_pairs_for_sentence.append( \
            #             self._get_tars_formatted_sentence(tag, sentence))
            #     for tag in sampled_tags_not_in_sentence:
            #         tag_text_pairs_for_sentence.append( \
            #             self._get_tars_formatted_sentence(tag, sentence))
            # else:
            for tag in all_tags:
                tag_text_pairs_for_sentence.append( \
                    self._get_tars_formatted_sentence(tag, sentence))
            tag_text_pairs.extend(tag_text_pairs_for_sentence)
        return tag_text_pairs

    # def _split_tag(self, tag: str):
    #     if "-" in tag:
    #         tag_split = tag.split("-")
    #         return tag_split[0], "-".join(tag_split[1:])
    #     else:
    #         return None, tag

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

        tag_dictionary = TARSSequenceTagger._make_ad_hoc_tag_dictionary(candidate_tag_set)

        # note current task
        existing_current_task = self.current_task

        # create a temporary task
        self.add_and_switch_to_new_task(TARSSequenceTagger.static_adhoc_task_identifier, tag_dictionary) #TODO: no tag_type needed here?

        try:
            # make zero shot predictions
            self.predict(sentences)
        except:
            log.error("Something went wrong during prediction. Ensure you pass Sentence objects.")
        finally:
            # switch to the pre-existing task
            self.switch_to_task(existing_current_task)
            self._drop_task(TARSSequenceTagger.static_adhoc_task_identifier)

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

                feature = self.forward(batch)

                if return_loss:
                    overall_loss += self._calculate_loss(feature, batch)

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

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence], sort=True
    ) -> torch.tensor:
        if isinstance(data_points, Sentence):
            data_points = [data_points]
        features = self.forward(data_points)
        return self._calculate_loss(features, data_points)

    def _transform_tars_scores(self, tars_scores, max_sequence_length: int):
        # M: num_classes in task, N: num_samples, L: max_sequence_length
        tars_scores = torch.nn.functional.softmax(tars_scores, dim=1)  # shape: N*L*M x 2
        scores = torch.reshape(tars_scores, (-1, len(self.tag_dictionary.item2idx), 2))  # shape: N*L x M x 2

        # choosing 1 instead of 0 here is not important, it is only important that always the same is seen as the true one
        # TODO: discuss whether the other one makes any sense or just the "YES" label could be enough
        target_scores = scores[:, :, 1]  # shape: N*L x M
        target_scores = torch.reshape(target_scores, (-1, max_sequence_length, len(self.tag_dictionary.item2idx))) # shape: N x L x M
        return target_scores

    def forward(self, sentences: List[Sentence]):
        # Transform input data into TARS format
        sentences = self._get_tars_formatted_sentences(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        pre_allocated_zero_tensor = torch.zeros(
            self.transformer_word_embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        # feed one tensor into linear layer instead of every item once
        all_embs = list()
        self.transformer_word_embeddings.embed(sentences)
        for sent in sentences:
            sep_token_reached = False
            offset = 0
            for tkn in sent:
                # nicht jedes Token zählen, sondern nur die nach erstem [SEP]
                if not sep_token_reached:
                    offset += 1
                    if tkn.text == self.transformer_word_embeddings.tokenizer.sep_token:
                        sep_token_reached = True
                    continue

                embed = tkn.get_embedding()
                all_embs.append(embed)

            nb_padding_tokens = longest_token_sequence_in_batch - len(sent)
            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[:self.transformer_word_embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch - offset,
                self.transformer_word_embeddings.embedding_length,
            ]
        )

        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        tag_scores = self.linear(sentence_tensor)
        transformed_scores = self._transform_tars_scores(tag_scores, longest_token_sequence_in_batch - offset)
        return transformed_scores

    def _calculate_loss(
            self, features: torch.tensor, sentences: List[Sentence]
    ) -> float:
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        tag_list: List = []
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.tensor(tag_idx, device=flair.device)
            tag_list.append(tag)
        score = 0
        for sentence_feats, sentence_tags, sentence_length in zip(
                features, tag_list, lengths
        ):
            sentence_feats = sentence_feats[:sentence_length]
            score += torch.nn.functional.cross_entropy(
                sentence_feats, sentence_tags, weight=self.loss_weights
            )
        score /= len(features)
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
