import logging
from pathlib import Path
from typing import List, Union, Callable, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import flair.nn
import flair.embeddings
from flair.data import Dictionary, Sentence, Label, Token, space_tokenizer, DataPoint
from flair.datasets import SentenceDataset, StringDataset
from flair.file_utils import cached_path
from flair.training_utils import (
    convert_labels_to_one_hot,
    Metric,
    Result,
    store_embeddings,
)

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

        text_embedding_list = [
            sentence.embedding.unsqueeze(0) for sentence in sentences
        ]
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)

        label_scores = self.decoder(text_embedding_tensor)

        return label_scores

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "document_embeddings": self.document_embeddings,
            "label_dictionary": self.label_dictionary,
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

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence, List[str], str],
        mini_batch_size: int = 32,
        embedding_storage_mode="none",
        multi_class_prob: bool = False,
        verbose: bool = False,
        use_tokenizer: Union[bool, Callable[[str], List[Token]]] = space_tokenizer,
    ) -> List[Sentence]:
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param embedding_storage_mode: 'none' for the minimum memory footprint, 'cpu' to store embeddings in Ram,
        'gpu' to store embeddings in GPU memory.
        :param multi_class_prob : return probability for all class for multiclass
        :param verbose: set to True to display a progress bar
        :param use_tokenizer: a custom tokenizer when string are provided (default is space based tokenizer).
        :return: the list of sentences containing the labels
        """
        predicted_label_type = self.label_type if self.label_type is not None else 'class'

        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, DataPoint) or isinstance(sentences, str):
                sentences = [sentences]

            if (flair.device.type == "cuda") and embedding_storage_mode == "cpu":
                log.warning(
                    "You are inferring on GPU with parameter 'embedding_storage_mode' set to 'cpu'."
                    "This option will slow down your inference, usually 'none' (default value) "
                    "is a better choice."
                )

            # filter empty sentences
            if isinstance(sentences[0], Sentence):
                sentences = [sentence for sentence in sentences if len(sentence) > 0]
            if len(sentences) == 0: return sentences

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )
            original_order_index = sorted(
                range(len(rev_order_len_index)), key=lambda k: rev_order_len_index[k]
            )

            reordered_sentences: List[Union[DataPoint, str]] = [
                sentences[index] for index in rev_order_len_index
            ]

            if isinstance(sentences[0], DataPoint):
                # remove previous embeddings
                store_embeddings(reordered_sentences, "none")
                dataset = SentenceDataset(reordered_sentences)
            else:
                dataset = StringDataset(
                    reordered_sentences, use_tokenizer=use_tokenizer
                )
            dataloader = DataLoader(
                dataset=dataset, batch_size=mini_batch_size, collate_fn=lambda x: x
            )

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            results: List[Sentence] = []
            for i, batch in enumerate(dataloader):
                if verbose:
                    dataloader.set_description(f"Inferencing on batch {i}")
                results += batch
                # stop if all sentences are empty
                if not batch:
                    continue

                scores = self.forward(batch)
                predicted_labels = self._obtain_labels(
                    scores, predict_prob=multi_class_prob
                )

                for (sentence, labels) in zip(batch, predicted_labels):
                    for label in labels:
                        sentence.add_label(predicted_label_type, label.value, label.score)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            results: List[Union[Sentence, str]] = [
                results[index] for index in original_order_index
            ]
            assert len(sentences) == len(results)
            return results

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embedding_storage_mode: str = "none",
    ) -> (Result, float):

        with torch.no_grad():
            eval_loss = 0

            metric = Metric("Evaluation", beta=self.beta)

            lines: List[str] = []
            batch_count: int = 0
            for batch in data_loader:

                batch_count += 1

                scores = self.forward(batch)
                predictions = self._obtain_labels(scores)
                loss = self._calculate_loss(scores, batch)

                eval_loss += loss

                sentences_for_batch = [sent.to_plain_string() for sent in batch]

                true_values_for_batch = [sentence.get_labels(self.label_type) for sentence in batch]
                available_labels = self.label_dictionary.get_items()

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

                    for label in available_labels:
                        if (
                            label in predictions_for_sentence
                            and label in true_values_for_sentence
                        ):
                            metric.add_tp(label)
                        elif (
                            label in predictions_for_sentence
                            and label not in true_values_for_sentence
                        ):
                            metric.add_fp(label)
                        elif (
                            label not in predictions_for_sentence
                            and label in true_values_for_sentence
                        ):
                            metric.add_fn(label)
                        elif (
                            label not in predictions_for_sentence
                            and label not in true_values_for_sentence
                        ):
                            metric.add_tn(label)

                store_embeddings(batch, embedding_storage_mode)

            eval_loss /= batch_count

            detailed_result = (
                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
            )
            for class_name in metric.get_classes():
                detailed_result += (
                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                    f"{metric.f_score(class_name):.4f}"
                )

            result = Result(
                main_score=metric.micro_avg_accuracy(),
                log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

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
        aws_resource_path = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.4"
        )

        model_map["de-offensive-language"] = "/".join(
            [
                aws_resource_path,
                "classy-offensive-de-rnn-cuda%3A0",
                "germ-eval-2018-task-1-v0.4.pt",
            ]
        )

        model_map["en-sentiment"] = "/".join(
            [aws_resource_path, "classy-imdb-en-rnn-cuda%3A0", "imdb-v0.4.pt"]
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
