from pathlib import Path
from typing import Union, Dict, List, Set, Optional, Tuple

import torch
from torch.utils.data import Dataset

import flair

from flair.data import Dictionary, Sentence, Label
from flair.datasets import SentenceDataset, DataLoader
from flair.file_utils import cached_path
from flair.models import SequenceTagger

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
import numpy as np

from tqdm import tqdm
import logging

from flair.models.text_classification_model import TARSClassifier
from flair.training_utils import Result, store_embeddings, Metric

log = logging.getLogger("flair")


class Switchable():

    def __init__(self):
        print('init Switchable')
        self._current_task = None
        self._task_specific_attributes = {}

    def get_current_tag_dictionary(self):
        return self._task_specific_attributes[self._current_task]['tag_dictionary']

    def get_current_tag_type(self):
        return self._task_specific_attributes[self._current_task]['tag_type']

    def add_and_switch_to_new_task(self,
                                   task_name,
                                   label_dictionary: Union[List, Set, Dictionary, str],
                                   tag_type: str = None,
                                   ):
        """
        Adds a new task to an existing TARS model. Sets necessary attributes and finally 'switches'
        to the new task. Parameters are similar to the constructor except for model choice, batch
        size and negative sampling. This method does not store the resultant model onto disk.
        :param task_name: a string depicting the name of the task
        :param label_dictionary: dictionary of the labels you want to predict
        :param multi_label: auto-detect if a corpus label dictionary is provided. Defaults to True otherwise
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        """
        if task_name in self._task_specific_attributes:
            log.warning("Task `%s` already exists in TARS model. Switching to it.", task_name)
        else:

            # make label dictionary if no Dictionary object is passed
            if isinstance(label_dictionary, Dictionary):
                label_dictionary = label_dictionary.get_items()

            # prepare dictionary of tags (without B- I- prefixes)
            tag_dictionary = Dictionary(add_unk=False)
            for tag in label_dictionary:
                if tag == 'O': continue
                if "-" in tag:
                    tag = tag.split("-")[1]
                    tag_dictionary.add_item(tag)
                else:
                    tag_dictionary.add_item(tag)

            self._task_specific_attributes[task_name] = {'tag_dictionary': tag_dictionary, 'tag_type': tag_type}

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
            log.error("Provided `%s` does not exist in the model. Consider calling "
                      "`add_and_switch_to_new_task` first.", task_name)
        else:
            self._current_task = task_name

    def _drop_task(self, task_name):
        if task_name in self._task_specific_attributes:
            if self._current_task == task_name:
                log.error("`%s` is the current task."
                          " Switch to some other task before dropping this.", task_name)
            else:
                self._task_specific_attributes.pop(task_name)
        else:
            log.warning("No task exists with the name `%s`.", task_name)


class TARSTagger(flair.nn.Model, Switchable):
    """
    TARS Sequence Tagger Model
    The model inherits TextClassifier class to provide usual interfaces such as evaluate,
    predict etc. It can encapsulate multiple tasks inside it. The user has to mention
    which task is intended to be used. In the backend, the model uses a BERT based binary
    text classifier which given a <label, text> pair predicts the probability of two classes
    "YES", and "NO". The input data is a usual Sentence object which is inflated
    by the model internally before pushing it through the transformer stack of BERT.
    """

    static_label_type = "tars_label"

    def __init__(
            self,
            task_name: str,
            tag_dictionary: Dictionary,
            tag_type: str,
            embeddings: str = 'bert-base-uncased',
            num_negative_labels_to_sample: int = 2,
            **tagger_args,
    ):
        """
        Initializes a TextClassifier
        :param task_name: a string depicting the name of the task
        :param label_dictionary: dictionary of labels you want to predict
        :param batch_size: batch size for forward pass while using BERT
        :param document_embeddings: name of the pre-trained transformer model e.g.,
        'bert-base-uncased' etc
        :num_negative_labels_to_sample: number of negative labels to sample for each
        positive labels against a sentence during training. Defaults to 2 negative
        labels for each positive label. The model would sample all the negative labels
        if None is passed. That slows down the training considerably.
        :param multi_label: auto-detected by default, but you can set this to True
        to force multi-label predictionor False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        :param beta: Parameter for F-beta score for evaluation and training annealing
        """

        flair.nn.Model.__init__(self)
        Switchable.__init__(self)

        from flair.embeddings import TransformerWordEmbeddings

        if not isinstance(embeddings, TransformerWordEmbeddings):
            embeddings = TransformerWordEmbeddings(model=embeddings,
                                                   fine_tune=True,
                                                   layers='-1',
                                                   layer_mean=False,
                                                   )

        # prepare TARS dictionary
        tars_dictionary = Dictionary(add_unk=False)
        tars_dictionary.add_item('O')
        tars_dictionary.add_item('S-')
        tars_dictionary.add_item('B-')
        tars_dictionary.add_item('E-')
        tars_dictionary.add_item('I-')

        # initialize a bare-bones sequence tagger
        self.tars_model = SequenceTagger(123,
                                         embeddings,
                                         tag_dictionary=tars_dictionary,
                                         tag_type=self.static_label_type,
                                         use_crf=False,
                                         use_rnn=False,
                                         reproject_embeddings=False,
                                         **tagger_args,
                                         )

        # transformer separator
        self.separator = str(self.tars_model.embeddings.tokenizer.sep_token)
        if self.tars_model.embeddings.tokenizer._bos_token:
            self.separator += str(self.tars_model.embeddings.tokenizer.bos_token)

        self.num_negative_labels_to_sample = num_negative_labels_to_sample
        self.label_nearest_map = None
        self.cleaned_up_labels = {}

        # Store task specific labels since TARS can handle multiple tasks
        self.add_and_switch_to_new_task(task_name, tag_dictionary, tag_type)

        self.beta = 1.

    def _compute_label_similarity_for_current_epoch(self):
        """
        Compute the similarity between all labels for better sampling of negatives
        """

        # get and embed all labels by making a Sentence object that contains only the label text
        all_labels = [label.decode("utf-8") for label in self.get_current_tag_dictionary().idx2item]
        label_sentences = [Sentence(label) for label in all_labels]

        self.tars_model.embeddings.eval()  # TODO: check if this is necessary
        self.tars_model.embeddings.embed(label_sentences)
        self.tars_model.embeddings.train()

        # print(label_sentences[0])
        # print(label_sentences[0][0])
        # print(label_sentences[0][0].get_embedding()[:10])

        # get each label embedding and scale between 0 and 1
        encodings_np = [sentence[0].get_embedding().cpu().detach().numpy() for sentence in label_sentences]
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
                    negative_label_probabilities[label][other_label] = \
                        similarity_matrix[row_index][column_index]
        self.label_nearest_map = negative_label_probabilities
        # print(self.label_nearest_map)

    def train(self, mode=True):
        """Populate label similarity map based on cosine similarity before running epoch

        If the `num_negative_labels_to_sample` is set to an integer value then before starting
        each epoch the model would create a similarity measure between the label names based
        on cosine distances between their BERT encoded embeddings.
        """
        if mode and self.num_negative_labels_to_sample is not None:
            self._compute_label_similarity_for_current_epoch()
            super(TARSTagger, self).train(mode)

        super(TARSTagger, self).train(mode)

    def _get_nearest_labels_for(self, labels):

        # print(labels)

        # if there are no labels, return a random sample as negatives
        if len(labels) == 0:
            tags = self.get_current_tag_dictionary().get_items()
            import random
            sample = random.sample(tags, k=self.num_negative_labels_to_sample)
            # print(sample)
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
            plausible_label_probabilities = np.array(plausible_label_probabilities, dtype='float64')
            plausible_label_probabilities += 1e-08
            plausible_label_probabilities /= np.sum(plausible_label_probabilities)

            # print(plausible_labels)
            # print(plausible_label_probabilities)

            if len(plausible_labels) > 0:
                num_samples = min(self.num_negative_labels_to_sample, len(plausible_labels))
                sampled_negative_labels = np.random.choice(plausible_labels,
                                                           num_samples,
                                                           replace=False,
                                                           p=plausible_label_probabilities)
                already_sampled_negative_labels.update(sampled_negative_labels)

        # negatives = []
        # negatives.extend(already_sampled_negative_labels)
        # negatives.sort()

        return already_sampled_negative_labels

    def _get_tars_formatted_sentence(self, label, sentence):

        original_text = sentence.to_tokenized_string()

        label_text_pair = " ".join([
            original_text,
            self.separator,
            label, ],
        )

        tars_sentence = Sentence(label_text_pair, use_tokenizer=False)

        for token in sentence:
            tag = token.get_tag(self.get_current_tag_type()).value

            if "-" in tag and tag.split('-')[1] == label:
                tars_tag = tag.split('-')[0] + '-'
            elif tag == label:
                tars_tag = "S-"
            else:
                tars_tag = "O"

            tars_sentence.get_token(token.idx).add_tag(self.static_label_type, tars_tag)

        return tars_sentence

    def _get_labels(self, sentence: Sentence) -> List[str]:
        labels = []
        for token in sentence:
            tag = token.get_tag(self.get_current_tag_type()).value
            if "-" in tag:
                tag = tag.split('-')[1]
                if tag not in labels:
                    labels.append(tag)
        return labels

    def _get_tars_formatted_sentences(self, sentences):
        label_text_pairs = []
        all_labels = [label.decode("utf-8") for label in self.get_current_tag_dictionary().idx2item]
        # print(all_labels)
        for sentence in sentences:
            label_text_pairs_for_sentence = []
            if self.training and self.num_negative_labels_to_sample is not None:
                positive_labels = self._get_labels(sentence)
                sampled_negative_labels = self._get_nearest_labels_for(positive_labels)

                for label in positive_labels:
                    label_text_pairs_for_sentence.append(self._get_tars_formatted_sentence(label, sentence))
                for label in sampled_negative_labels:
                    label_text_pairs_for_sentence.append(self._get_tars_formatted_sentence(label, sentence))

                # if len(positive_labels) == 0:
                #     for label in all_labels:
                #         label_text_pairs_for_sentence.append(self._get_tars_formatted_sentence(label, sentence))

            else:
                for label in all_labels:
                    label_text_pairs_for_sentence.append(self._get_tars_formatted_sentence(label, sentence))
            label_text_pairs.extend(label_text_pairs_for_sentence)

        # if len(label_text_pairs) == 0:
        #     randomly_sampled = np.random.choice(all_labels, 2, replace=False)
        #     for label in randomly_sampled:
        #         label_text_pairs.append(self._get_tars_formatted_sentence(label, sentence))

        return label_text_pairs

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),

            "current_task": self._current_task,
            "tag_type": self.get_current_tag_type(),
            "tag_dictionary": self.get_current_tag_dictionary(),
            "tars_model": self.tars_model,
            "num_negative_labels_to_sample": self.num_negative_labels_to_sample,

            "task_specific_attributes": self._task_specific_attributes,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        print("init TARS")

        # init new TARS classifier
        model = TARSTagger(
            task_name=state["current_task"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            embeddings=state["tars_model"].embeddings,
            num_negative_labels_to_sample=state["num_negative_labels_to_sample"],
        )
        # set all task information
        model.task_specific_attributes = state["task_specific_attributes"]
        # linear layers of internal classifier
        model.load_state_dict(state["state_dict"])
        return model

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence]
    ) -> torch.tensor:

        if type(data_points) == Sentence:
            data_points = [data_points]

        # Transform input data into TARS format
        sentences = self._get_tars_formatted_sentences(data_points)

        loss = self.tars_model.forward_loss(sentences)

        return loss

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences

    @staticmethod
    def _fetch_model(model_name) -> str:

        model_map = {}
        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"

        model_map["tars-base"] = "/".join([hu_path, "tars-base", "tars-base-v8.pt"])

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name

    def evaluate(
            self,
            sentences: Union[List[Sentence], Dataset],
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
            wsd_evaluation: bool = False,
            **kwargs,
    ) -> (Result, float):

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        eval_loss = 0
        eval_count = 0

        batch_no: int = 0

        metric = Metric("Evaluation", beta=self.beta)

        lines: List[str] = []

        y_true = []
        y_pred = []

        for batch in data_loader:

            # predict for batch
            loss_and_count = self.predict(batch,
                                          embedding_storage_mode=embedding_storage_mode,
                                          mini_batch_size=mini_batch_size,
                                          label_name='predicted',
                                          return_loss=True)

            eval_loss += loss_and_count[0]
            eval_count += loss_and_count[1]
            batch_no += 1

            for sentence in batch:

                # make list of gold tags
                gold_spans = sentence.get_spans(self.get_current_tag_type())
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

        return result, eval_loss / eval_count

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            mini_batch_size=32,
            verbose: bool = False,
            label_name: Optional[str] = None,
            return_loss=False,
            embedding_storage_mode="none",
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
        if label_name == None:
            label_name = self.get_current_tag_type()

        # with torch.no_grad():
        if not sentences:
            return sentences

        if isinstance(sentences, Sentence):
            sentences = [sentences]

        # set context if not set already
        previous_sentence = None
        for sentence in sentences:
            if sentence.is_context_set(): continue
            sentence._previous_sentence = previous_sentence
            sentence._next_sentence = None
            if previous_sentence: previous_sentence._next_sentence = sentence
            previous_sentence = sentence

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
        overall_count = 0
        batch_no = 0
        with torch.no_grad():
            for batch in dataloader:

                batch_no += 1

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {batch_no}")

                batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                # go through each sentence in the batch
                for sentence in batch:

                    # always remove tags first
                    for token in sentence:
                        token.remove_labels(label_name)

                    all_labels = [label.decode("utf-8") for label in self.get_current_tag_dictionary().idx2item]

                    all_detected = {}
                    for label in all_labels:
                        tars_sentence = self._get_tars_formatted_sentence(label, sentence)

                        loss_and_count = self.tars_model.predict(tars_sentence,
                                                                 label_name=label_name,
                                                                 all_tag_prob=True,
                                                                 return_loss=True)
                        overall_loss += loss_and_count[0].item()
                        overall_count += loss_and_count[1]

                        for span in tars_sentence.get_spans(label_name):
                            span.set_label('tars_temp_label', label)
                            all_detected[span] = span.score

                        for span in tars_sentence.get_spans(label_name):
                            for token in span:
                                corresponding_token = sentence.get_token(token.idx)
                                if corresponding_token is None: continue
                                if corresponding_token.get_tag(label_name).value != '' and \
                                        corresponding_token.get_tag(label_name).score > token.get_tag(label_name).score:
                                    continue
                                corresponding_token.add_tag(
                                    label_name,
                                    token.get_tag(label_name).value + label,
                                    token.get_tag(label_name).score,
                                )

                    # import operator
                    # sorted_x = sorted(all_detected.items(), key=operator.itemgetter(1))
                    # sorted_x.reverse()
                    # print(sorted_x)
                    # for tuple in sorted_x:
                    #     span = tuple[0]
                    #
                    #     tag_this = True
                    #
                    # for token in span:
                    #     corresponding_token = sentence.get_token(token.idx)
                    #     if corresponding_token is None:
                    #         tag_this = False
                    #         continue
                    #     if corresponding_token.get_tag(label_name).value != '' and \
                    #             corresponding_token.get_tag(label_name).score > token.get_tag(label_name).score:
                    #         tag_this = False
                    #         continue
                    #
                    # if tag_this:
                    #     for token in span:
                    #         corresponding_token = sentence.get_token(token.idx)
                    #         corresponding_token.add_tag(
                    #             label_name,
                    #             token.get_tag(label_name).value + span.get_labels('tars_temp_label')[0].value,
                    #             token.get_tag(label_name).score,
                    #         )

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

        if return_loss:
            return overall_loss, overall_count

    def predict_zero_shot(self,
                          sentences: Union[List[Sentence], Sentence],
                          candidate_label_set: Union[List[str], Set[str], str],
                          multi_label: bool = True):
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

        label_dictionary = TARSClassifier._make_ad_hoc_label_dictionary(candidate_label_set, multi_label)

        # note current task
        existing_current_task = self.current_task

        # create a temporary task
        self.add_and_switch_to_new_task(TARSClassifier.static_adhoc_task_identifier,
                                        label_dictionary)

        try:
            # make zero shot predictions
            self.predict(sentences)
        except:
            log.error("Something went wrong during prediction. Ensure you pass Sentence objects.")

        finally:
            # switch to the pre-existing task
            self.switch_to_task(existing_current_task)

            self._drop_task(TARSClassifier.static_adhoc_task_identifier)

        return
