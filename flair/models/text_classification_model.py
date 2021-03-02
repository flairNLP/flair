import logging
from pathlib import Path
from typing import List, Union, Dict, Optional, Set

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import numpy as np

import sklearn.metrics as metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
import flair.nn
import flair.embeddings
from flair.data import Dictionary, Sentence, Label, DataPoint, DataPair
from flair.datasets import SentenceDataset, DataLoader
from flair.file_utils import cached_path
from flair.training_utils import convert_labels_to_one_hot, Result, store_embeddings

log = logging.getLogger("flair")


class TextClassifier(flair.nn.Model):
    """
    Text Classification Model
    The model takes word embeddings, puts them into an RNN to obtain a text representation, and puts the
    text representation in the end into a linear layer to get the actual class label.
    The model can handle single and multi class data sets.
    """

    def __init__(
            self,
            document_embeddings: flair.embeddings.DocumentEmbeddings,
            label_dictionary: Dictionary,
            label_type: str = None,
            multi_label: bool = None,
            multi_label_threshold: float = 0.5,
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
    ):
        """
        Initializes a TextClassifier
        :param document_embeddings: embeddings used to embed each data point
        :param label_dictionary: dictionary of labels you want to predict
        :param multi_label: auto-detected by default, but you can set this to True to force multi-label prediction
        or False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        """

        super(TextClassifier, self).__init__()

        self.document_embeddings: flair.embeddings.DocumentRNNEmbeddings = document_embeddings
        self.label_dictionary: Dictionary = label_dictionary
        self.label_type = label_type

        if multi_label is not None:
            self.multi_label = multi_label
        else:
            self.multi_label = self.label_dictionary.multi_label

        self.multi_label_threshold = multi_label_threshold

        self.beta = beta

        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.label_dictionary)
            weight_list = [1. for i in range(n_classes)]
            for i, tag in enumerate(self.label_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        self.decoder = nn.Linear(
            self.document_embeddings.embedding_length, len(self.label_dictionary)
        )

        nn.init.xavier_uniform_(self.decoder.weight)

        if self.multi_label:
            self.loss_function = nn.BCEWithLogitsLoss(weight=self.loss_weights)
        else:
            self.loss_function = nn.CrossEntropyLoss(weight=self.loss_weights)

        # auto-spawn on GPU if available
        self.to(flair.device)

    def forward(self, sentences):

        self.document_embeddings.embed(sentences)

        embedding_names = self.document_embeddings.get_names()

        text_embedding_list = [
            sentence.get_embedding(embedding_names).unsqueeze(0) for sentence in sentences
        ]
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)

        label_scores = self.decoder(text_embedding_tensor)

        return label_scores

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "document_embeddings": self.document_embeddings,
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
            "multi_label": self.multi_label,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        label_type = None if "label_type" not in state.keys() else state["label_type"]

        model = TextClassifier(
            document_embeddings=state["document_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=label_type,
            multi_label=state["multi_label"],
            beta=beta,
            loss_weights=weights,
        )

        model.load_state_dict(state["state_dict"])
        return model

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence]
    ) -> torch.tensor:

        scores = self.forward(data_points)

        return self._calculate_loss(scores, data_points)

    def _calculate_loss(self, scores, data_points):

        labels = self._labels_to_one_hot(data_points) if self.multi_label \
            else self._labels_to_indices(data_points)

        return self.loss_function(scores, labels)

    def _forward_scores_and_loss(
            self, data_points: Union[List[Sentence], Sentence], return_loss=False):
        scores = self.forward(data_points)

        loss = None
        if return_loss:
            loss = self._calculate_loss(scores, data_points)

        return scores, loss

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            mini_batch_size: int = 32,
            multi_class_prob: bool = False,
            verbose: bool = False,
            label_name: Optional[str] = None,
            return_loss=False,
            embedding_storage_mode="none",
    ):
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param multi_class_prob : return probability for all class for multiclass
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if label_name == None:
            label_name = self.label_type if self.label_type is not None else 'label'

        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, DataPoint):
                sentences = [sentences]

            # filter empty sentences
            if isinstance(sentences[0], DataPoint):
                sentences = [sentence for sentence in sentences if len(sentence) > 0]
            if len(sentences) == 0: return sentences

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )

            reordered_sentences: List[Union[DataPoint, str]] = [
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

                # stop if all sentences are empty
                if not batch:
                    continue

                scores, loss = self._forward_scores_and_loss(batch, return_loss)

                if return_loss:
                    overall_loss += loss

                predicted_labels = self._obtain_labels(scores, predict_prob=multi_class_prob)

                for (sentence, labels) in zip(batch, predicted_labels):
                    for label in labels:
                        if self.multi_label or multi_class_prob:
                            sentence.add_label(label_name, label.value, label.score)
                        else:
                            sentence.set_label(label_name, label.value, label.score)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss / batch_no

    def evaluate(
            self,
            sentences: Union[List[DataPoint], Dataset],
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
    ) -> (Result, float):

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        # use scikit-learn to evaluate
        y_true = []
        y_pred = []

        with torch.no_grad():
            eval_loss = 0

            lines: List[str] = []
            batch_count: int = 0

            for batch in data_loader:
                batch_count += 1

                # remove previously predicted labels
                [sentence.remove_labels('predicted') for sentence in batch]

                # get the gold labels
                true_values_for_batch = [sentence.get_labels(self.label_type) for sentence in batch]

                # predict for batch
                loss = self.predict(batch,
                                    embedding_storage_mode=embedding_storage_mode,
                                    mini_batch_size=mini_batch_size,
                                    label_name='predicted',
                                    return_loss=True)

                eval_loss += loss

                sentences_for_batch = [sent.to_plain_string() for sent in batch]

                # get the predicted labels
                predictions = [sentence.get_labels('predicted') for sentence in batch]

                for sentence, prediction, true_value in zip(
                        sentences_for_batch,
                        predictions,
                        true_values_for_batch,
                ):
                    eval_line = "{}\t{}\t{}\n".format(
                        sentence, true_value, prediction
                    )
                    lines.append(eval_line)

                for predictions_for_sentence, true_values_for_sentence in zip(
                        predictions, true_values_for_batch
                ):

                    true_values_for_sentence = [label.value for label in true_values_for_sentence]
                    predictions_for_sentence = [label.value for label in predictions_for_sentence]

                    y_true_instance = np.zeros(len(self.label_dictionary), dtype=int)
                    for i in range(len(self.label_dictionary)):
                        if self.label_dictionary.get_item_for_index(i) in true_values_for_sentence:
                            y_true_instance[i] = 1
                    y_true.append(y_true_instance.tolist())

                    y_pred_instance = np.zeros(len(self.label_dictionary), dtype=int)
                    for i in range(len(self.label_dictionary)):
                        if self.label_dictionary.get_item_for_index(i) in predictions_for_sentence:
                            y_pred_instance[i] = 1
                    y_pred.append(y_pred_instance.tolist())

                store_embeddings(batch, embedding_storage_mode)

            # remove predicted labels
            for sentence in sentences:
                sentence.annotation_layers['predicted'] = []

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make "classification report"
            target_names = []
            for i in range(len(self.label_dictionary)):
                target_names.append(self.label_dictionary.get_item_for_index(i))
            classification_report = metrics.classification_report(y_true, y_pred, digits=4,
                                                                  target_names=target_names, zero_division=0)

            # get scores
            micro_f_score = round(metrics.fbeta_score(y_true, y_pred, beta=self.beta, average='micro', zero_division=0),
                                  4)
            accuracy_score = round(metrics.accuracy_score(y_true, y_pred), 4)
            macro_f_score = round(metrics.fbeta_score(y_true, y_pred, beta=self.beta, average='macro', zero_division=0),
                                  4)
            precision_score = round(metrics.precision_score(y_true, y_pred, average='macro', zero_division=0), 4)
            recall_score = round(metrics.recall_score(y_true, y_pred, average='macro', zero_division=0), 4)

            detailed_result = (
                    "\nResults:"
                    f"\n- F-score (micro) {micro_f_score}"
                    f"\n- F-score (macro) {macro_f_score}"
                    f"\n- Accuracy {accuracy_score}"
                    '\n\nBy class:\n' + classification_report
            )

            # line for log file
            if not self.multi_label:
                log_header = "ACCURACY"
                log_line = f"\t{accuracy_score}"
            else:
                log_header = "PRECISION\tRECALL\tF1\tACCURACY"
                log_line = f"{precision_score}\t" \
                           f"{recall_score}\t" \
                           f"{macro_f_score}\t" \
                           f"{accuracy_score}"

            result = Result(
                main_score=micro_f_score,
                log_line=log_line,
                log_header=log_header,
                detailed_results=detailed_result,
            )

            eval_loss /= batch_count

            return result, eval_loss

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                "Ignore {} sentence(s) with no tokens.".format(
                    len(sentences) - len(filtered_sentences)
                )
            )
        return filtered_sentences

    def _obtain_labels(
            self, scores: List[List[float]], predict_prob: bool = False
    ) -> List[List[Label]]:
        """
        Predicts the labels of sentences.
        :param scores: the prediction scores from the model
        :return: list of predicted labels
        """

        if self.multi_label:
            return [self._get_multi_label(s) for s in scores]

        elif predict_prob:
            return [self._predict_label_prob(s) for s in scores]

        return [self._get_single_label(s) for s in scores]

    def _get_multi_label(self, label_scores) -> List[Label]:
        labels = []

        sigmoid = torch.nn.Sigmoid()

        results = list(map(lambda x: sigmoid(x), label_scores))
        for idx, conf in enumerate(results):
            if conf > self.multi_label_threshold:
                label = self.label_dictionary.get_item_for_index(idx)
                labels.append(Label(label, conf.item()))

        return labels

    def _get_single_label(self, label_scores) -> List[Label]:
        softmax = torch.nn.functional.softmax(label_scores, dim=0)
        conf, idx = torch.max(softmax, 0)
        label = self.label_dictionary.get_item_for_index(idx.item())

        return [Label(label, conf.item())]

    def _predict_label_prob(self, label_scores) -> List[Label]:
        softmax = torch.nn.functional.softmax(label_scores, dim=0)
        label_probs = []
        for idx, conf in enumerate(softmax):
            label = self.label_dictionary.get_item_for_index(idx)
            label_probs.append(Label(label, conf.item()))
        return label_probs

    def _labels_to_one_hot(self, sentences: List[Sentence]):

        label_list = []
        for sentence in sentences:
            label_list.append([label.value for label in sentence.get_labels(self.label_type)])

        one_hot = convert_labels_to_one_hot(label_list, self.label_dictionary)
        one_hot = [torch.FloatTensor(l).unsqueeze(0) for l in one_hot]
        one_hot = torch.cat(one_hot, 0).to(flair.device)
        return one_hot

    def _labels_to_indices(self, sentences: List[Sentence]):

        indices = [
            torch.LongTensor(
                [
                    self.label_dictionary.get_idx_for_item(label.value)
                    for label in sentence.get_labels(self.label_type)
                ]
            )
            for sentence in sentences
        ]

        vec = torch.cat(indices, 0).to(flair.device)

        return vec

    @staticmethod
    def _fetch_model(model_name) -> str:

        model_map = {}
        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"

        model_map["de-offensive-language"] = "/".join(
            [hu_path, "de-offensive-language", "germ-eval-2018-task-1-v0.5.pt"]
        )

        # English sentiment models
        model_map["sentiment"] = "/".join(
            [hu_path, "sentiment-curated-distilbert", "sentiment-en-mix-distillbert_4.pt"]
        )
        model_map["en-sentiment"] = "/".join(
            [hu_path, "sentiment-curated-distilbert", "sentiment-en-mix-distillbert_4.pt"]
        )
        model_map["sentiment-fast"] = "/".join(
            [hu_path, "sentiment-curated-fasttext-rnn", "sentiment-en-mix-ft-rnn.pt"]
        )

        # Communicative Functions Model
        model_map["communicative-functions"] = "/".join(
            [hu_path, "comfunc", "communicative-functions.pt"]
        )

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name

    def __str__(self):
        return super(flair.nn.Model, self).__str__().rstrip(')') + \
               f'  (beta): {self.beta}\n' + \
               f'  (weights): {self.weight_dict}\n' + \
               f'  (weight_tensor) {self.loss_weights}\n)'


class TextPairClassifier(TextClassifier):
    """
    Text Pair Classification Model for tasks such as Recognizing Textual Entailment, build upon TextClassifier.
    The model takes document embeddings and puts resulting text representation(s) into a linear layer to get the 
    actual class label. We provide two ways to embed the DataPairs: Either by embedding both DataPoints
    and concatenating the resulting vectors ("embed_separately=True") or by concatenating the DataPoints and embedding
    the resulting vector ("embed_separately=False").
    """

    def __init__(
            self,
            document_embeddings: flair.embeddings.DocumentEmbeddings,
            label_dictionary: Dictionary,
            embed_separately: bool = False,
            label_type: str = None,
            multi_label: bool = None,
            multi_label_threshold: float = 0.5,
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
    ):
        """
        :param document_embeddings: embeddings used to embed the Datapairs
        :param label_dictionary: dictionary of labels you want to predict
        :param label_type: name of the label
        :param embed_separately: If True, the model embeds both data points separately, else cross-embedding
        :param multi_label: auto-detected by default, but you can set this to True to force multi-label prediction
        or False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        """

        self.bi_mode = embed_separately
        # Initialize TextClassifier
        super(TextPairClassifier, self).__init__(document_embeddings,
                                                 label_dictionary,
                                                 label_type=label_type,
                                                 multi_label=multi_label,
                                                 multi_label_threshold=multi_label_threshold,
                                                 beta=beta,
                                                 loss_weights=loss_weights)

        # if bi_mode == True the linear layer needs twice the length of the embeddings as input size
        # since we concatenate the embeddings of the two DataPoints in the DataPairs
        if self.bi_mode:
            self.decoder = nn.Linear(
                2 * self.document_embeddings.embedding_length, len(self.label_dictionary)
            ).to(flair.device)

            nn.init.xavier_uniform_(self.decoder.weight)

    def _get_state_dict(self):
        model_state = super()._get_state_dict()
        model_state["bi_mode"] = self.bi_mode
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        label_type = None if "label_type" not in state.keys() else state["label_type"]
        mode = True if "bi_mode" not in state.keys() else state["bi_mode"]

        model = TextPairClassifier(
            document_embeddings=state["document_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=label_type,
            multi_label=state["multi_label"],
            beta=beta,
            loss_weights=weights,
            embed_separately=mode
        )

        model.load_state_dict(state["state_dict"])
        return model

    def forward(self, datapairs):

        embedding_names = self.document_embeddings.get_names()

        if isinstance(datapairs, DataPair):
            datapairs = [datapairs]

        if self.bi_mode:  # embed both sentences seperately, concatenate the resulting vectors
            first_elements = [pair.first for pair in datapairs]
            second_elements = [pair.second for pair in datapairs]

            self.document_embeddings.embed(first_elements)

            self.document_embeddings.embed(second_elements)

            text_embedding_list = [
                torch.cat([a.get_embedding(embedding_names), b.get_embedding(embedding_names)], 0).unsqueeze(0)
                for (a, b) in zip(first_elements, second_elements)
            ]

        else:  # concatenate the sentences and embed together

            # TODO: Transformers use special separator symbols in the beginning and between elements
            #      of datapair. Here should be a case dinstintion between the different transformers.
            if isinstance(self.document_embeddings, flair.embeddings.document.TransformerDocumentEmbeddings):
                sep = '[SEP]'
            else:
                sep = ' '

            concatenated_sentences = [Sentence(pair.first.to_plain_string() + sep + pair.second.to_plain_string()) for
                                      pair in datapairs]

            self.document_embeddings.embed(concatenated_sentences)

            text_embedding_list = [
                sentence.get_embedding(embedding_names).unsqueeze(0) for sentence in concatenated_sentences
            ]

        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)

        # linear layer
        label_scores = self.decoder(text_embedding_tensor)

        return label_scores


class TARSClassifier(TextClassifier):
    """
    TARS Classification Model
    The model inherits TextClassifier class to provide usual interfaces such as evaluate,
    predict etc. It can encapsulate multiple tasks inside it. The user has to mention
    which task is intended to be used. In the backend, the model uses a BERT based binary
    text classifier which given a <label, text> pair predicts the probability of two classes
    "YES", and "NO". The input data is a usual Sentence object which is inflated
    by the model internally before pushing it through the transformer stack of BERT.
    """

    static_label_yes = "YES"
    static_label_no = "NO"
    static_label_type = "tars_label"
    static_adhoc_task_identifier = "adhoc_dummy"

    def __init__(
            self,
            task_name: str,
            label_dictionary: Dictionary,
            batch_size: int = 16,
            document_embeddings: str = 'bert-base-uncased',
            num_negative_labels_to_sample: int = 2,
            label_type: str = None,
            multi_label: bool = None,
            multi_label_threshold: float = 0.5,
            beta: float = 1.0
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
        from flair.embeddings.document import TransformerDocumentEmbeddings

        if not isinstance(document_embeddings, TransformerDocumentEmbeddings):
            document_embeddings = TransformerDocumentEmbeddings(model=document_embeddings,
                                                                fine_tune=True,
                                                                batch_size=batch_size)

        super(TARSClassifier, self).__init__(document_embeddings,
                                             label_dictionary,
                                             label_type=label_type,
                                             multi_label=multi_label,
                                             multi_label_threshold=multi_label_threshold,
                                             beta=beta)

        # Drop unnecessary attributes from Parent class
        self.document_embeddings = None
        self.decoder = None
        self.loss_function = None

        # prepare binary label dictionary
        tars_label_dictionary = Dictionary(add_unk=False)
        tars_label_dictionary.add_item(self.static_label_no)
        tars_label_dictionary.add_item(self.static_label_yes)

        self.tars_model = TextClassifier(document_embeddings,
                                         tars_label_dictionary,
                                         label_type=self.static_label_type,
                                         multi_label=False,
                                         beta=1.0,
                                         loss_weights=None)

        self.num_negative_labels_to_sample = num_negative_labels_to_sample
        self.label_nearest_map = None
        self.cleaned_up_labels = {}
        self.current_task = None

        # Store task specific labels since TARS can handle multiple tasks
        self.task_specific_attributes = {}
        self.add_and_switch_to_new_task(task_name, label_dictionary, multi_label,
                                        multi_label_threshold, label_type, beta)

    def add_and_switch_to_new_task(self,
                                   task_name,
                                   label_dictionary: Union[List, Set, Dictionary, str],
                                   multi_label: bool = True,
                                   multi_label_threshold: float = 0.5,
                                   label_type: str = None,
                                   beta: float = 1.0
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
        if task_name in self.task_specific_attributes:
            log.warning("Task `%s` already exists in TARS model. Switching to it.", task_name)
        else:

            # make label dictionary if no Dictionary object is passed
            if isinstance(label_dictionary, (list, set, str)):
                label_dictionary = TARSClassifier._make_ad_hoc_label_dictionary(label_dictionary, multi_label)

            self.task_specific_attributes[task_name] = {}
            self.task_specific_attributes[task_name]['label_dictionary'] = label_dictionary
            self.task_specific_attributes[task_name]['multi_label'] = label_dictionary.multi_label
            self.task_specific_attributes[task_name]['multi_label_threshold'] = multi_label_threshold
            self.task_specific_attributes[task_name]['label_type'] = label_type
            self.task_specific_attributes[task_name]['beta'] = beta

        self.switch_to_task(task_name)

    def list_existing_tasks(self) -> Set[str]:
        """
        Lists existing tasks in the loaded TARS model on the console.
        """
        return set(self.task_specific_attributes.keys())

    def _get_cleaned_up_label(self, label):
        """
        Does some basic clean up of the provided labels, stores them, looks them up.
        """
        if label not in self.cleaned_up_labels:
            self.cleaned_up_labels[label] = label.replace("_", " ")
        return self.cleaned_up_labels[label]

    def _compute_label_similarity_for_current_epoch(self):
        """
        Compute the similarity between all labels for better sampling of negatives
        """

        # get and embed all labels by making a Sentence object that contains only the label text
        all_labels = [label.decode("utf-8") for label in self.label_dictionary.idx2item]
        label_sentences = [Sentence(self._get_cleaned_up_label(label)) for label in all_labels]
        self.tars_model.document_embeddings.embed(label_sentences)

        # get each label embedding and scale between 0 and 1
        encodings_np = [sentence.get_embedding().cpu().detach().numpy() for \
                        sentence in label_sentences]
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

    def train(self, mode=True):
        """Populate label similarity map based on cosine similarity before running epoch

        If the `num_negative_labels_to_sample` is set to an integer value then before starting
        each epoch the model would create a similarity measure between the label names based
        on cosine distances between their BERT encoded embeddings.
        """
        if mode and self.num_negative_labels_to_sample is not None:
            self._compute_label_similarity_for_current_epoch()
            super(TARSClassifier, self).train(mode)

        super(TARSClassifier, self).train(mode)

    def _get_nearest_labels_for(self, labels):
        already_sampled_negative_labels = set()

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

            if len(plausible_labels) > 0:
                num_samples = min(self.num_negative_labels_to_sample, len(plausible_labels))
                sampled_negative_labels = np.random.choice(plausible_labels,
                                                           num_samples,
                                                           replace=False,
                                                           p=plausible_label_probabilities)
                already_sampled_negative_labels.update(sampled_negative_labels)

        return already_sampled_negative_labels

    def _get_tars_formatted_sentence(self, label, original_text, tars_label=None):
        label_text_pair = " ".join([self._get_cleaned_up_label(label),
                                    self.tars_model.document_embeddings.tokenizer.sep_token,
                                    original_text])
        label_text_pair_sentence = Sentence(label_text_pair, use_tokenizer=False)
        if tars_label is not None:
            if tars_label:
                label_text_pair_sentence.add_label(self.tars_model.label_type,
                                                   TARSClassifier.static_label_yes)
            else:
                label_text_pair_sentence.add_label(self.tars_model.label_type,
                                                   TARSClassifier.static_label_no)
        return label_text_pair_sentence

    def _get_tars_formatted_sentences(self, sentences):
        label_text_pairs = []
        all_labels = [label.decode("utf-8") for label in self.label_dictionary.idx2item]
        for sentence in sentences:
            original_text = sentence.to_tokenized_string()
            label_text_pairs_for_sentence = []
            if self.training and self.num_negative_labels_to_sample is not None:
                positive_labels = {label.value for label in sentence.get_labels()}
                sampled_negative_labels = self._get_nearest_labels_for(positive_labels)
                for label in positive_labels:
                    label_text_pairs_for_sentence.append( \
                        self._get_tars_formatted_sentence(label, original_text, True))
                for label in sampled_negative_labels:
                    label_text_pairs_for_sentence.append( \
                        self._get_tars_formatted_sentence(label, original_text, False))
            else:
                positive_labels = {label.value for label in sentence.get_labels()}
                for label in all_labels:
                    tars_label = None if len(positive_labels) == 0 else label in positive_labels
                    label_text_pairs_for_sentence.append( \
                        self._get_tars_formatted_sentence(label, original_text, tars_label))
            label_text_pairs.extend(label_text_pairs_for_sentence)
        return label_text_pairs

    def switch_to_task(self, task_name):
        """
        Switches to a task which was previously added.
        """
        if task_name not in self.task_specific_attributes:
            log.error("Provided `%s` does not exist in the model. Consider calling "
                      "`add_and_switch_to_new_task` first.", task_name)
        else:
            self.current_task = task_name
            self.multi_label = self.task_specific_attributes[task_name]['multi_label']
            self.multi_label_threshold = \
                self.task_specific_attributes[task_name]['multi_label_threshold']
            self.label_dictionary = self.task_specific_attributes[task_name]['label_dictionary']
            self.label_type = self.task_specific_attributes[task_name]['label_type']
            self.beta = self.task_specific_attributes[task_name]['beta']

    def _get_state_dict(self):
        model_state = super(TARSClassifier, self)._get_state_dict()
        model_state.update({
            "current_task": self.current_task,
            "task_specific_attributes": self.task_specific_attributes,
            "tars_model": self.tars_model,
            "num_negative_labels_to_sample": self.num_negative_labels_to_sample
        })
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        task_name = state["current_task"]
        print("init TARS")
        
        # init new TARS classifier
        model = TARSClassifier(
            task_name,
            label_dictionary = state["task_specific_attributes"][task_name]['label_dictionary'],
            document_embeddings=state["tars_model"].document_embeddings,
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
        # Transform input data into TARS format
        sentences = self._get_tars_formatted_sentences(data_points)

        return self.tars_model.forward_loss(sentences)

    def _transform_tars_scores(self, tars_scores):
        # M: num_classes in task, N: num_samples
        # reshape scores MN x 2 -> N x M x 2
        # import torch
        # a = torch.arange(30)
        # b = torch.reshape(-1, 3, 2)
        # c = b[:,:,1]
        tars_scores = torch.nn.functional.softmax(tars_scores, dim=1)
        scores = torch.reshape(tars_scores, (-1, len(self.label_dictionary.item2idx), 2))

        # target shape N x M
        target_scores = scores[:, :, 1]
        return target_scores

    def _forward_scores_and_loss(
            self, data_points: Union[List[Sentence], Sentence], return_loss=False):
        transformed_sentences = self._get_tars_formatted_sentences(data_points)
        label_scores = self.tars_model.forward(transformed_sentences)
        # Transform label_scores
        transformed_scores = self._transform_tars_scores(label_scores)

        loss = None
        if return_loss:
            loss = self.tars_model._calculate_loss(label_scores, transformed_sentences)

        return transformed_scores, loss

    def forward(self, sentences):
        transformed_sentences = self._get_tars_formatted_sentences(sentences)
        label_scores = self.tars_model.forward(transformed_sentences)

        # Transform label_scores into current task's desired format
        transformed_scores = self._transform_tars_scores(label_scores)

        return transformed_scores

    def _get_multi_label(self, label_scores) -> List[Label]:
        labels = []

        for idx, conf in enumerate(label_scores):
            if conf > self.multi_label_threshold:
                label = self.label_dictionary.get_item_for_index(idx)
                labels.append(Label(label, conf.item()))

        return labels

    def _get_single_label(self, label_scores) -> List[Label]:
        conf, idx = torch.max(label_scores, 0)
        # TARS does not do a softmax, so confidence of the best predicted class might be very low.
        # Therefore enforce a min confidence of 0.5 for a match.
        label = self.label_dictionary.get_item_for_index(idx.item())
        return [Label(label, conf.item())]

    @staticmethod
    def _make_ad_hoc_label_dictionary(candidate_label_set: Union[List[str], Set[str], str],
                                      multi_label: bool = True) -> Dictionary:
        """
        Creates a dictionary given a set of candidate labels
        :return: dictionary of labels
        """
        label_dictionary: Dictionary = Dictionary(add_unk=False)
        label_dictionary.multi_label = multi_label

        # make list if only one candidate label is passed
        if isinstance(candidate_label_set, str):
            candidate_label_set = {candidate_label_set}

        # if list is passed, convert to set
        if not isinstance(candidate_label_set, set):
            candidate_label_set = set(candidate_label_set)

        for label in candidate_label_set:
            label_dictionary.add_item(label)

        return label_dictionary

    def _drop_task(self, task_name):
        if task_name in self.task_specific_attributes:
            if self.current_task == task_name:
                log.error("`%s` is the current task."
                          " Switch to some other task before dropping this.", task_name)
            else:
                self.task_specific_attributes.pop(task_name)
        else:
            log.warning("No task exists with the name `%s`.", task_name)

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
                                        label_dictionary, multi_label)

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

    def predict_all_tasks(self, sentences: Union[List[Sentence], Sentence]):

        # remember current task
        existing_current_task = self.current_task

        # predict with each task model
        for task in self.list_existing_tasks():
            self.switch_to_task(task)
            self.predict(sentences, label_name=task)

        # switch to the pre-existing task
        self.switch_to_task(existing_current_task)

    @staticmethod
    def _fetch_model(model_name) -> str:

        model_map = {}
        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"

        model_map["tars-base"] = "/".join([hu_path, "tars-base", "tars-base-v8.pt"])

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name
