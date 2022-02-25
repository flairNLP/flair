import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

import flair
from flair.data import Dictionary, Sentence, Span
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import (
    TokenEmbeddings,
    TransformerDocumentEmbeddings,
    TransformerWordEmbeddings,
)
from flair.file_utils import cached_path
from flair.models.sequence_tagger_model import SequenceTagger
from flair.models.text_classification_model import TextClassifier
from flair.training_utils import store_embeddings

log = logging.getLogger("flair")


class FewshotClassifier(flair.nn.Classifier[Sentence]):
    def __init__(self):
        self._current_task = None
        self._task_specific_attributes = {}
        self.label_nearest_map = None
        self.tars_model: flair.nn.Classifier[Sentence]

        super(FewshotClassifier, self).__init__()

    def forward_loss(
        self, data_points: Union[List[Sentence], Sentence]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:

        if not isinstance(data_points, list):
            data_points = [data_points]

        # Transform input data into TARS format
        sentences = self._get_tars_formatted_sentences(data_points)

        loss = self.tars_model.forward_loss(sentences)
        return loss

    @property
    def tars_embeddings(self):
        raise NotImplementedError

    def _get_tars_formatted_sentence(self, label, sentence):
        raise NotImplementedError

    def _get_tars_formatted_sentences(self, sentences: List[Sentence]):
        label_text_pairs = []
        all_labels = [label.decode("utf-8") for label in self.get_current_label_dictionary().idx2item]
        for sentence in sentences:
            label_text_pairs_for_sentence = []
            if self.training and self.num_negative_labels_to_sample is not None:

                positive_labels = list(
                    OrderedDict.fromkeys([label.value for label in sentence.get_labels(self.label_type)])
                )

                sampled_negative_labels = self._get_nearest_labels_for(positive_labels)

                for label in positive_labels:
                    label_text_pairs_for_sentence.append(self._get_tars_formatted_sentence(label, sentence))
                for label in sampled_negative_labels:
                    label_text_pairs_for_sentence.append(self._get_tars_formatted_sentence(label, sentence))

            else:
                for label in all_labels:
                    label_text_pairs_for_sentence.append(self._get_tars_formatted_sentence(label, sentence))
            label_text_pairs.extend(label_text_pairs_for_sentence)

        return label_text_pairs

    def _get_nearest_labels_for(self, labels):

        # if there are no labels, return a random sample as negatives
        if len(labels) == 0:
            tags = self.get_current_label_dictionary().get_items()
            import random

            sample = random.sample(tags, k=self.num_negative_labels_to_sample)
            return sample

        already_sampled_negative_labels = set()

        # otherwise, go through all labels
        for label in labels:

            plausible_labels = []
            plausible_label_probabilities = []
            for plausible_label in self.label_nearest_map[label]:
                if plausible_label in already_sampled_negative_labels or plausible_label in labels:
                    continue
                else:
                    plausible_labels.append(plausible_label)
                    plausible_label_probabilities.append(self.label_nearest_map[label][plausible_label])

            # make sure the probabilities always sum up to 1
            plausible_label_probabilities = np.array(plausible_label_probabilities, dtype="float64")
            plausible_label_probabilities += 1e-08
            plausible_label_probabilities /= np.sum(plausible_label_probabilities)

            if len(plausible_labels) > 0:
                num_samples = min(self.num_negative_labels_to_sample, len(plausible_labels))
                sampled_negative_labels = np.random.choice(
                    plausible_labels,
                    num_samples,
                    replace=False,
                    p=plausible_label_probabilities,
                )
                already_sampled_negative_labels.update(sampled_negative_labels)

        return already_sampled_negative_labels

    def train(self, mode=True):
        """Populate label similarity map based on cosine similarity before running epoch

        If the `num_negative_labels_to_sample` is set to an integer value then before starting
        each epoch the model would create a similarity measure between the label names based
        on cosine distances between their BERT encoded embeddings.
        """
        if mode and self.num_negative_labels_to_sample is not None:
            self._compute_label_similarity_for_current_epoch()
            super().train(mode)

        super().train(mode)

    def _compute_label_similarity_for_current_epoch(self):
        """
        Compute the similarity between all labels for better sampling of negatives
        """

        # get and embed all labels by making a Sentence object that contains only the label text
        all_labels = [label.decode("utf-8") for label in self.get_current_label_dictionary().idx2item]
        label_sentences = [Sentence(label) for label in all_labels]

        self.tars_embeddings.eval()  # TODO: check if this is necessary
        self.tars_embeddings.embed(label_sentences)
        self.tars_embeddings.train()

        # get each label embedding and scale between 0 and 1
        if isinstance(self.tars_embeddings, TokenEmbeddings):
            encodings_np = [sentence[0].get_embedding().cpu().detach().numpy() for sentence in label_sentences]
        else:
            encodings_np = [sentence.get_embedding().cpu().detach().numpy() for sentence in label_sentences]

        normalized_encoding = minmax_scale(encodings_np)

        # compute similarity matrix
        similarity_matrix = cosine_similarity(normalized_encoding)

        # the higher the similarity, the greater the chance that a label is
        # sampled as negative example
        negative_label_probabilities = {}
        for row_index, label in enumerate(all_labels):
            negative_label_probabilities[label] = {}
            for column_index, other_label in enumerate(all_labels):
                if label != other_label:
                    negative_label_probabilities[label][other_label] = similarity_matrix[row_index][column_index]
        self.label_nearest_map = negative_label_probabilities

    def get_current_label_dictionary(self):
        label_dictionary = self._task_specific_attributes[self._current_task]["label_dictionary"]
        return label_dictionary

    def get_current_label_type(self):
        return self._task_specific_attributes[self._current_task]["label_type"]

    def is_current_task_multi_label(self):
        return self._task_specific_attributes[self._current_task]["multi_label"]

    def add_and_switch_to_new_task(
        self,
        task_name,
        label_dictionary: Union[List, Set, Dictionary, str],
        label_type: str,
        multi_label: bool = True,
        force_switch: bool = False,
    ):
        """
        Adds a new task to an existing TARS model. Sets necessary attributes and finally 'switches'
        to the new task. Parameters are similar to the constructor except for model choice, batch
        size and negative sampling. This method does not store the resultant model onto disk.
        :param task_name: a string depicting the name of the task
        :param label_dictionary: dictionary of the labels you want to predict
        :param label_type: string to identify the label type ('ner', 'sentiment', etc.)
        :param multi_label: whether this task is a multi-label prediction problem
        :param force_switch: if True, will overwrite existing task with same name
        """
        if task_name in self._task_specific_attributes and not force_switch:
            log.warning("Task `%s` already exists in TARS model. Switching to it.", task_name)
        else:
            # make label dictionary if no Dictionary object is passed
            if isinstance(label_dictionary, Dictionary):
                label_dictionary = label_dictionary.get_items()
            if type(label_dictionary) == str:
                label_dictionary = [label_dictionary]

            # prepare dictionary of tags (without B- I- prefixes and without UNK)
            tag_dictionary = Dictionary(add_unk=False)
            for tag in label_dictionary:
                if tag == "<unk>" or tag == "O":
                    continue
                if tag[1] == "-":
                    tag = tag[2:]
                    tag_dictionary.add_item(tag)
                else:
                    tag_dictionary.add_item(tag)

            self._task_specific_attributes[task_name] = {
                "label_dictionary": tag_dictionary,
                "label_type": label_type,
                "multi_label": multi_label,
            }

        self.switch_to_task(task_name)

    def list_existing_tasks(self) -> Set[str]:
        """
        Lists existing tasks in the loaded TARS model on the console.
        """
        return set(self._task_specific_attributes.keys())

    def switch_to_task(self, task_name):
        """
        Switches to a task which was previously added.
        """
        if task_name not in self._task_specific_attributes:
            log.error(
                "Provided `%s` does not exist in the model. Consider calling " "`add_and_switch_to_new_task` first.",
                task_name,
            )
        else:
            self._current_task = task_name

    def _drop_task(self, task_name):
        if task_name in self._task_specific_attributes:
            if self._current_task == task_name:
                log.error(
                    "`%s` is the current task." " Switch to some other task before dropping this.",
                    task_name,
                )
            else:
                self._task_specific_attributes.pop(task_name)
        else:
            log.warning("No task exists with the name `%s`.", task_name)

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens.")
        return filtered_sentences

    @property
    def label_type(self):
        return self.get_current_label_type()

    def predict_zero_shot(
        self,
        sentences: Union[List[Sentence], Sentence],
        candidate_label_set: Union[List[str], Set[str], str],
        multi_label: bool = True,
    ):
        """
        Method to make zero shot predictions from the TARS model
        :param sentences: input sentence objects to classify
        :param candidate_label_set: set of candidate labels
        :param multi_label: indicates whether multi-label or single class prediction. Defaults to True.
        """

        # check if candidate_label_set is empty
        if candidate_label_set is None or len(candidate_label_set) == 0:
            log.warning("Provided candidate_label_set is empty")
            return

        # make list if only one candidate label is passed
        if isinstance(candidate_label_set, str):
            candidate_label_set = {candidate_label_set}

        # create label dictionary
        label_dictionary = Dictionary(add_unk=False)
        for label in candidate_label_set:
            label_dictionary.add_item(label)

        # note current task
        existing_current_task = self._current_task

        # create a temporary task
        self.add_and_switch_to_new_task(
            task_name="ZeroShot",
            label_dictionary=label_dictionary,
            label_type="-".join(label_dictionary.get_items()),
            multi_label=multi_label,
        )

        try:
            # make zero shot predictions
            self.predict(sentences)
        finally:
            # switch to the pre-existing task
            self.switch_to_task(existing_current_task)
            self._drop_task("ZeroShot")

        return


class TARSTagger(FewshotClassifier):
    """
    TARS model for sequence tagging. In the backend, the model uses a BERT based 5-class
    sequence labeler which given a <label, text> pair predicts the probability for each word
    to belong to one of the BIOES classes. The input data is a usual Sentence object which is inflated
    by the model internally before pushing it through the transformer stack of BERT.
    """

    static_label_type = "tars_label"

    def __init__(
        self,
        task_name: Optional[str] = None,
        label_dictionary: Optional[Dictionary] = None,
        label_type: Optional[str] = None,
        embeddings: Union[TransformerWordEmbeddings, str] = "bert-base-uncased",
        num_negative_labels_to_sample: int = 2,
        prefix: bool = True,
        **tagger_args,
    ):
        """
        Initializes a TextClassifier
        :param task_name: a string depicting the name of the task
        :param label_dictionary: dictionary of labels you want to predict
        :param embeddings: name of the pre-trained transformer model e.g.,
        'bert-base-uncased' etc
        :param num_negative_labels_to_sample: number of negative labels to sample for each
        positive labels against a sentence during training. Defaults to 2 negative
        labels for each positive label. The model would sample all the negative labels
        if None is passed. That slows down the training considerably.
        """
        super(TARSTagger, self).__init__()

        if isinstance(embeddings, str):
            embeddings = TransformerWordEmbeddings(
                model=embeddings,
                fine_tune=True,
                layers="-1",
                layer_mean=False,
            )

        # prepare TARS dictionary
        tars_dictionary = Dictionary(add_unk=False)
        tars_dictionary.add_item("entity")
        tars_dictionary.span_labels = True

        # initialize a bare-bones sequence tagger
        self.tars_model: SequenceTagger = SequenceTagger(
            hidden_size=123,
            embeddings=embeddings,
            tag_dictionary=tars_dictionary,
            tag_type=self.static_label_type,
            use_crf=False,
            use_rnn=False,
            reproject_embeddings=False,
            **tagger_args,
        )

        # transformer separator
        self.separator = str(self.tars_embeddings.tokenizer.sep_token)
        if self.tars_embeddings.tokenizer._bos_token:
            self.separator += str(self.tars_embeddings.tokenizer.bos_token)

        self.prefix = prefix
        self.num_negative_labels_to_sample = num_negative_labels_to_sample

        if task_name and label_dictionary and label_type:
            # Store task specific labels since TARS can handle multiple tasks
            self.add_and_switch_to_new_task(task_name, label_dictionary, label_type)
        else:
            log.info(
                "TARS initialized without a task. You need to call .add_and_switch_to_new_task() "
                "before training this model"
            )

    def _get_tars_formatted_sentence(self, label, sentence):

        original_text = sentence.to_tokenized_string()

        label_text_pair = (
            f"{label} {self.separator} {original_text}" if self.prefix else f"{original_text} {self.separator} {label}"
        )

        label_length = 0 if not self.prefix else len(label.split(" ")) + len(self.separator.split(" "))

        # make a tars sentence where all labels are O by default
        tars_sentence = Sentence(label_text_pair, use_tokenizer=False)

        for entity_label in sentence.get_labels(self.label_type):
            if entity_label.value == label:
                new_span = Span(
                    [tars_sentence.get_token(token.idx + label_length) for token in entity_label.data_point]
                )
                new_span.add_label(self.static_label_type, value="entity")

        return tars_sentence

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "current_task": self._current_task,
            "tag_type": self.get_current_label_type(),
            "tag_dictionary": self.get_current_label_dictionary(),
            "tars_model": self.tars_model,
            "num_negative_labels_to_sample": self.num_negative_labels_to_sample,
            "prefix": self.prefix,
            "task_specific_attributes": self._task_specific_attributes,
        }
        return model_state

    @staticmethod
    def _fetch_model(model_name) -> str:

        if model_name == "tars-ner":
            cache_dir = Path("models")
            model_name = cached_path(
                "https://nlp.informatik.hu-berlin.de/resources/models/tars-ner/tars-ner.pt",
                cache_dir=cache_dir,
            )

        return model_name

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        # init new TARS classifier
        model = super()._init_model_with_state_dict(
            state,
            task_name=state["current_task"],
            label_dictionary=state["tag_dictionary"],
            label_type=state["tag_type"],
            embeddings=state["tars_model"].embeddings,
            num_negative_labels_to_sample=state["num_negative_labels_to_sample"],
            prefix=state["prefix"],
            **kwargs,
        )
        # set all task information
        model._task_specific_attributes = state["task_specific_attributes"]

        return model

    @property
    def tars_embeddings(self):
        return self.tars_model.embeddings

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size=32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
        most_probable_first: bool = True,
    ):
        # return
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
            label_name = self.get_current_label_type()

        # with torch.no_grad():
        if not sentences:
            return sentences

        if not isinstance(sentences, list):
            sentences = [sentences]

        reordered_sentences = sorted(sentences, key=lambda s: len(s), reverse=True)

        dataloader = DataLoader(
            dataset=FlairDatapointDataset(reordered_sentences),
            batch_size=mini_batch_size,
        )

        # progress bar for verbosity
        if verbose:
            dataloader = tqdm(dataloader)

        overall_loss = 0
        overall_count = 0
        with torch.no_grad():
            for batch in dataloader:

                batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                # go through each sentence in the batch
                for sentence in batch:

                    # always remove tags first
                    sentence.remove_labels(label_name)

                    all_labels = [label.decode("utf-8") for label in self.get_current_label_dictionary().idx2item]

                    all_detected = {}
                    for label in all_labels:
                        tars_sentence = self._get_tars_formatted_sentence(label, sentence)

                        loss_and_count = self.tars_model.predict(
                            tars_sentence,
                            label_name=label_name,
                            return_loss=True,
                        )

                        overall_loss += loss_and_count[0].item()
                        overall_count += loss_and_count[1]

                        for predicted in tars_sentence.get_labels(label_name):
                            predicted.value = label
                            all_detected[predicted] = predicted.score

                    if most_probable_first:
                        import operator

                        already_set_indices: List[int] = []

                        sorted_x = sorted(all_detected.items(), key=operator.itemgetter(1))
                        sorted_x.reverse()
                        for tuple in sorted_x:
                            # get the span and its label
                            label = tuple[0]
                            # label = span.get_labels("tars_temp_label")[0].value
                            label_length = (
                                0 if not self.prefix else len(label.value.split(" ")) + len(self.separator.split(" "))
                            )

                            # determine whether tokens in this span already have a label
                            tag_this = True
                            for token in label.data_point:
                                corresponding_token = sentence.get_token(token.idx - label_length)
                                if corresponding_token is None:
                                    tag_this = False
                                    continue
                                if token.idx in already_set_indices:
                                    tag_this = False
                                    continue

                            # only add if all tokens have no label
                            if tag_this:
                                already_set_indices.extend(token.idx for token in label.data_point)
                                predicted_span = Span(
                                    [sentence.get_token(token.idx - label_length) for token in label.data_point]
                                )
                                predicted_span.add_label(label_name, value=label.value, score=label.score)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

        if return_loss:
            return overall_loss, overall_count

    def _print_predictions(self, batch, gold_label_type):

        lines = []
        if self.tars_model.predict_spans:
            for datapoint in batch:
                # all labels default to "O"
                for token in datapoint:
                    token.set_label("gold_bio", "O")
                    token.set_label("predicted_bio", "O")

                # set gold token-level
                for gold_label in datapoint.get_labels(gold_label_type):
                    gold_span: Span = gold_label.data_point
                    prefix = "B-"
                    for token in gold_span:
                        token.set_label("gold_bio", prefix + gold_label.value)
                        prefix = "I-"

                # set predicted token-level
                for predicted_label in datapoint.get_labels("predicted"):
                    predicted_span: Span = predicted_label.data_point
                    prefix = "B-"
                    for token in predicted_span:
                        token.set_label("predicted_bio", prefix + predicted_label.value)
                        prefix = "I-"

                # now print labels in CoNLL format
                for token in datapoint:
                    eval_line = (
                        f"{token.text} "
                        f"{token.get_label('gold_bio').value} "
                        f"{token.get_label('predicted_bio').value}\n"
                    )
                    lines.append(eval_line)
                lines.append("\n")
        return lines


class TARSClassifier(FewshotClassifier):
    """
    TARS model for text classification. In the backend, the model uses a BERT based binary
    text classifier which given a <label, text> pair predicts the probability of two classes
    "True", and "False". The input data is a usual Sentence object which is inflated
    by the model internally before pushing it through the transformer stack of BERT.
    """

    static_label_type = "tars_label"
    LABEL_MATCH = "YES"
    LABEL_NO_MATCH = "NO"

    def __init__(
        self,
        task_name: Optional[str] = None,
        label_dictionary: Optional[Dictionary] = None,
        label_type: Optional[str] = None,
        embeddings: Union[TransformerDocumentEmbeddings, str] = "bert-base-uncased",
        num_negative_labels_to_sample: int = 2,
        prefix: bool = True,
        **tagger_args,
    ):
        """
        Initializes a TextClassifier
        :param task_name: a string depicting the name of the task
        :param label_dictionary: dictionary of labels you want to predict
        :param embeddings: name of the pre-trained transformer model e.g.,
        'bert-base-uncased' etc
        :param num_negative_labels_to_sample: number of negative labels to sample for each
        positive labels against a sentence during training. Defaults to 2 negative
        labels for each positive label. The model would sample all the negative labels
        if None is passed. That slows down the training considerably.
        :param multi_label: auto-detected by default, but you can set this to True
        to force multi-label predictionor False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        :param beta: Parameter for F-beta score for evaluation and training annealing
        """
        super(TARSClassifier, self).__init__()

        if isinstance(embeddings, str):
            embeddings = TransformerDocumentEmbeddings(
                model=embeddings,
                fine_tune=True,
                layers="-1",
                layer_mean=False,
            )

        # prepare TARS dictionary
        tars_dictionary = Dictionary(add_unk=False)
        tars_dictionary.add_item(self.LABEL_NO_MATCH)
        tars_dictionary.add_item(self.LABEL_MATCH)

        # initialize a bare-bones sequence tagger
        self.tars_model = TextClassifier(
            document_embeddings=embeddings,
            label_dictionary=tars_dictionary,
            label_type=self.static_label_type,
            **tagger_args,
        )

        # transformer separator
        self.separator = str(self.tars_embeddings.tokenizer.sep_token)
        if self.tars_embeddings.tokenizer._bos_token:
            self.separator += str(self.tars_embeddings.tokenizer.bos_token)

        self.prefix = prefix
        self.num_negative_labels_to_sample = num_negative_labels_to_sample

        if task_name and label_dictionary and label_type:
            # Store task specific labels since TARS can handle multiple tasks
            self.add_and_switch_to_new_task(task_name, label_dictionary, label_type)
        else:
            log.info(
                "TARS initialized without a task. You need to call .add_and_switch_to_new_task() "
                "before training this model"
            )

        self.clean_up_labels = True

    def _clean(self, label_value: str) -> str:
        if self.clean_up_labels:
            return label_value.replace("_", " ")
        else:
            return label_value

    def _get_tars_formatted_sentence(self, label, sentence):

        label = self._clean(label)

        original_text = sentence.to_tokenized_string()

        label_text_pair = (
            f"{label} {self.separator} {original_text}" if self.prefix else f"{original_text} {self.separator} {label}"
        )

        sentence_labels = [self._clean(label.value) for label in sentence.get_labels(self.get_current_label_type())]

        tars_label = self.LABEL_MATCH if label in sentence_labels else self.LABEL_NO_MATCH

        tars_sentence = Sentence(label_text_pair, use_tokenizer=False).add_label(self.static_label_type, tars_label)

        return tars_sentence

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "current_task": self._current_task,
            "label_type": self.get_current_label_type(),
            "label_dictionary": self.get_current_label_dictionary(),
            "tars_model": self.tars_model,
            "num_negative_labels_to_sample": self.num_negative_labels_to_sample,
            "task_specific_attributes": self._task_specific_attributes,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        # init new TARS classifier
        label_dictionary = state["label_dictionary"]
        label_type = "default_label" if not state["label_type"] else state["label_type"]

        model: TARSClassifier = super()._init_model_with_state_dict(
            state,
            task_name=state["current_task"],
            label_dictionary=label_dictionary,
            label_type=label_type,
            embeddings=state["tars_model"].document_embeddings,
            num_negative_labels_to_sample=state["num_negative_labels_to_sample"],
            **kwargs,
        )

        # set all task information
        model._task_specific_attributes = state["task_specific_attributes"]

        return model

    @staticmethod
    def _fetch_model(model_name) -> str:

        model_map = {}
        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"

        model_map["tars-base"] = "/".join([hu_path, "tars-base", "tars-base-v8.pt"])

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name

    @property
    def tars_embeddings(self):
        return self.tars_model.document_embeddings

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size=32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
        label_threshold: float = 0.5,
        multi_label: Optional[bool] = None,
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
            label_name = self.get_current_label_type()

        if multi_label is None:
            multi_label = self.is_current_task_multi_label()

        # with torch.no_grad():
        if not sentences:
            return sentences

        if isinstance(sentences, Sentence):
            sentences = [sentences]

        # set context if not set already
        previous_sentence = None
        for sentence in sentences:
            if sentence.is_context_set():
                continue
            sentence._previous_sentence = previous_sentence
            sentence._next_sentence = None
            if previous_sentence:
                previous_sentence._next_sentence = sentence
            previous_sentence = sentence

        reordered_sentences = sorted(sentences, key=lambda s: len(s), reverse=True)

        dataloader = DataLoader(
            dataset=FlairDatapointDataset(reordered_sentences),
            batch_size=mini_batch_size,
        )

        # progress bar for verbosity
        if verbose:
            progressbar = tqdm(dataloader)
            progressbar.set_description("Batch inference")
            dataloader = progressbar

        overall_loss = 0
        overall_count = 0
        batch_no = 0
        with torch.no_grad():
            for batch in dataloader:

                batch_no += 1

                batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                # go through each sentence in the batch
                for sentence in batch:

                    # always remove tags first
                    sentence.remove_labels(label_name)

                    all_labels = [label.decode("utf-8") for label in self.get_current_label_dictionary().idx2item]

                    best_label = None
                    for label in all_labels:
                        tars_sentence = self._get_tars_formatted_sentence(label, sentence)

                        loss_and_count = self.tars_model.predict(
                            tars_sentence,
                            label_name=label_name,
                            return_loss=True,
                            return_probabilities_for_all_classes=True if label_threshold < 0.5 else False,
                        )

                        overall_loss += loss_and_count[0].item()
                        overall_count += loss_and_count[1]

                        # add all labels that according to TARS match the text and are above threshold
                        for predicted_tars_label in tars_sentence.get_labels(label_name):
                            if (
                                predicted_tars_label.value == self.LABEL_MATCH
                                and predicted_tars_label.score > label_threshold
                            ):
                                # do not add labels below confidence threshold
                                sentence.add_label(label_name, label, predicted_tars_label.score)

                    # only use label with highest confidence if enforcing single-label predictions
                    if not multi_label:
                        if len(sentence.get_labels(label_name)) > 0:
                            # get all label scores and do an argmax to get the best label
                            label_scores = torch.tensor(
                                [label.score for label in sentence.get_labels(label_name)],
                                dtype=torch.float,
                            )
                            best_label = sentence.get_labels(label_name)[torch.argmax(label_scores)]

                            # remove previously added labels and only add the best label
                            sentence.remove_labels(label_name)
                            sentence.add_label(
                                typename=label_name,
                                value=best_label.value,
                                score=best_label.score,
                            )

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

        if return_loss:
            return overall_loss, overall_count
