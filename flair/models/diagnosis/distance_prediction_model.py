import logging
from math import floor
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataset import Dataset

import flair.embeddings
import flair.nn
from flair.data import Label, Sentence, _iter_dataset
from flair.training_utils import MetricRegression, Result, store_embeddings

log = logging.getLogger("flair")


class WeightedMSELoss(_Loss):
    def __init__(
        self,
        regr_loss_step: float,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(WeightedMSELoss, self).__init__(size_average, reduce, reduction)
        self.regr_loss_step = regr_loss_step

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1 + self.regr_loss_step * target

        return (weight * ((predictions - target) ** 2)).mean()


class DistancePredictor(flair.nn.Model[Sentence]):
    """
    DistancePredictor
    Model to predict distance between two words given their embeddings, modeled either as a classification or a
    regression model. Takes (contextual) word embedding as input.
    The pair of word embeddings is passed through a linear layer that predicts their distance in a sentence.
    Note: When used for training the batch size must be set to 1!!!
    """

    def __init__(
        self,
        word_embeddings: flair.embeddings.TokenEmbeddings,
        max_distance: int = 20,
        beta: float = 1.0,
        loss_max_weight: float = 1,
        regression=False,
        regr_loss_step: float = 0,
    ):
        """
        Initializes a DistClassifier
        :param word_embeddings: embeddings used to embed each sentence
        .param max_distance: max dist between word pairs = number of predicted classes - 1
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_max_weight: Only for classification: Since small distances between word pairs occur mor frequent it makes sense to give them less weight
        in the loss function. loss_max_weight will be used as the weight for the maximum distance and should be a number >=1
        The other weights decrease with equidistant steps from high to low distance.
        :param regression: if True the class does regression instead of classification
        :param regr_loss_step: if > 0, the MSE-Loss in regression will be weighted. Word pairs with
        distance 0 have weight 1. Then, as the distance increases, the weight in the loss function,
        increases step by step with size regr_loss_step
        """

        super(DistancePredictor, self).__init__()

        self.word_embeddings: flair.embeddings.TokenEmbeddings = word_embeddings

        self.beta = beta

        self.loss_max_weight = loss_max_weight

        self.regression = regression

        self.regr_loss_step = regr_loss_step

        if not regression:
            self.max_distance = max_distance

            # weights for loss function
            if self.loss_max_weight > 1:
                step = (self.loss_max_weight - 1) / self.max_distance

                weight_list = [1.0 + i * step for i in range(self.max_distance + 1)]

                self.loss_weights: Optional[torch.Tensor] = torch.FloatTensor(weight_list).to(flair.device)

            else:
                self.loss_weights = None

            # iput size is two times wordembedding size since we use pair of words as input
            # the output size is max_distance + 1, i.e. we allow 0,1,...,max_distance words between pairs
            self.decoder = nn.Linear(self.word_embeddings.embedding_length * 2, self.max_distance + 1)

            self.loss_function: _Loss = nn.CrossEntropyLoss(weight=self.loss_weights)

        # regression
        else:
            self.max_distance = 1000000

            # input size is two times word embedding size since we use pair of words as input
            # the output size is 1
            self.decoder = nn.Linear(self.word_embeddings.embedding_length * 2, 1)

            if regr_loss_step > 0:
                self.loss_function = WeightedMSELoss(regr_loss_step)
            else:
                self.loss_function = nn.MSELoss()

        nn.init.xavier_uniform_(self.decoder.weight)

        # auto-spawn on GPU if available
        self.to(flair.device)

    def label_type(self):
        return "distance"

    # forward allows only a single sentcence!!
    def forward(self, sentence: Sentence):

        # embed words of sentence
        self.word_embeddings.embed(sentence)

        # go through all pairs of words with a maximum number of max_distance in between
        numberOfWords = len(sentence)
        text_embedding_list = []
        # go through all pairs
        for i in range(numberOfWords):
            for j in range(i + 1, min(i + self.max_distance + 2, numberOfWords)):
                text_embedding_list.append(torch.cat((sentence[i].embedding, sentence[j].embedding)).unsqueeze(0))

        # 2-dim matrix whose rows are the embeddings of word pairs of the sentence
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)

        label_scores = self.decoder(text_embedding_tensor)

        if self.regression:
            return label_scores.squeeze(1)

        return label_scores

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "word_embeddings": self.word_embeddings,
            "max_distance": self.max_distance,
            "beta": self.beta,
            "loss_max_weight": self.loss_max_weight,
            "regression": self.regression,
            "regr_loss_step": self.regr_loss_step,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):

        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weight = 1 if "loss_max_weight" not in state.keys() else state["loss_max_weight"]

        return super()._init_model_with_state_dict(
            state,
            word_embeddings=state["word_embeddings"],
            max_distance=state["max_distance"],
            beta=beta,
            loss_max_weight=weight,
            regression=state["regression"],
            regr_loss_step=state["regr_loss_step"],
            **kwargs,
        )

    # So far only one sentence allowed
    # If list of sentences is handed the function works with the first sentence of the list
    def forward_loss(self, data_points: Union[List[Sentence], Sentence]) -> torch.Tensor:

        if isinstance(data_points, list):  # first sentence
            data_points = data_points[0]

        if len(data_points) < 2:
            return torch.tensor([0.0], requires_grad=True)

        scores = self.forward(data_points)

        return self._calculate_loss(scores, data_points)

    # Assume data_points is a single sentence!!!
    # scores are the predictions for each word pair
    def _calculate_loss(self, scores, data_points):

        indices = []
        numberOfWords = len(data_points)

        # classification needs labels to be integers, regression needs labels to be float
        # this is due to the different loss functions
        if not self.regression:
            for i in range(numberOfWords):
                for j in range(i + 1, min(i + self.max_distance + 2, numberOfWords)):
                    indices.append(torch.LongTensor([j - i - 1]))  # distance between words
        else:
            for i in range(numberOfWords):
                for j in range(i + 1, min(i + self.max_distance + 2, numberOfWords)):
                    indices.append(torch.Tensor([j - i - 1]))  # distance between words

        labels = torch.cat(indices, 0).to(flair.device)

        return self.loss_function(scores, labels)

    # only single sentences as input
    def _forward_scores_and_loss(self, data_points: Union[List[Sentence], Sentence], return_loss=False):

        if isinstance(data_points, list):  # first sentence
            data_points = data_points[0]

        scores = self.forward(data_points)

        loss = None
        if return_loss:
            loss = self._calculate_loss(scores, data_points)

        return scores, loss

    def evaluate(
        self,
        data_points: Union[List[Sentence], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 1,  # unnecessary, but trainer.train calls evaluate with this parameter
        num_workers: Optional[int] = 8,
        **kwargs,
    ) -> Result:

        if isinstance(data_points, Dataset):
            data_points = list(_iter_dataset(data_points))

        if self.regression:
            return self.evaluate_regression(
                sentences=data_points,
                out_path=out_path,
                embedding_storage_mode=embedding_storage_mode,
            )

        return self.evaluate_classification(
            sentences=data_points,
            out_path=out_path,
            embedding_storage_mode=embedding_storage_mode,
        )

    def evaluate_regression(
        self,
        sentences: List[Sentence],
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
    ) -> Result:

        with torch.no_grad():

            buckets = [0 for _ in range(11)]

            eval_loss = torch.zeros(1, device=flair.device)

            metric = MetricRegression("Evaluation")

            lines: List[str] = []

            max_dist_plus_one = max([len(sent) for sent in sentences]) - 1

            num_occurences = [0 for _ in range(max_dist_plus_one)]

            cumulated_values = [0 for _ in range(max_dist_plus_one)]

            for sentence in sentences:

                if len(sentence) < 2:  # we need at least 2 words per sentence
                    continue

                scores, loss = self._forward_scores_and_loss(sentence, return_loss=True)

                predictions = scores.tolist()

                # gold labels
                true_values_for_sentence = []
                numberOfPairs = 0
                numberOfWords = len(sentence)
                lines.append(sentence.to_tokenized_string() + "\n")
                for i in range(numberOfWords):
                    for j in range(i + 1, min(i + self.max_distance + 2, numberOfWords)):
                        true_dist = j - i - 1
                        pred = predictions[numberOfPairs]

                        true_values_for_sentence.append(true_dist)

                        # for output text file
                        eval_line = f"({i},{j})\t{true_dist}\t{pred:.2f}\n"
                        lines.append(eval_line)

                        # for buckets
                        error = abs(true_dist - pred)
                        if error >= 10:
                            buckets[10] += 1
                        else:
                            buckets[floor(error)] += 1

                        # for average prediction
                        num_occurences[true_dist] += 1
                        cumulated_values[true_dist] += pred

                        numberOfPairs += 1

                eval_loss += loss / numberOfPairs

                metric.true.extend(true_values_for_sentence)
                metric.pred.extend(predictions)

                store_embeddings([sentence], embedding_storage_mode)

            eval_loss /= len(sentences)  # w.r.t self.loss

            # add some statistics to the output
            eval_line = f"Number of Sentences: {len(sentences)}\nBuckets:\n | 0-1 | 1-2 | 2-3 | 3-4 | 4-5 | 5-6 | 6-7 | 7-8 | 8-9 | 9-10 | >10 |\n"
            lines.append(eval_line)
            eval_line = "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                buckets[0],
                buckets[1],
                buckets[2],
                buckets[3],
                buckets[4],
                buckets[5],
                buckets[6],
                buckets[7],
                buckets[8],
                buckets[9],
                buckets[10],
            )
            lines.append(eval_line)
            lines.append("\nAverage predicted values per distance:\n")
            eval_line = ""
            for i in range(max_dist_plus_one):
                eval_line += str(i) + ": " + f"{cumulated_values[i] / num_occurences[i]:.2f}" + " "
                if i != 0 and i % 15 == 0:
                    eval_line += "\n"

            lines.append(eval_line)

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            log_line = f"{metric.mean_squared_error()}\t{metric.spearmanr()}\t{metric.pearsonr()}"
            log_header = "MSE\tSPEARMAN\tPEARSON"

            detailed_result = (
                f"AVG: mse: {metric.mean_squared_error():.4f} - "
                f"mae: {metric.mean_absolute_error():.4f} - "
                f"pearson: {metric.pearsonr():.4f} - "
                f"spearman: {metric.spearmanr():.4f}"
            )

            result: Result = Result(metric.pearsonr(), log_header, log_line, detailed_result, loss=eval_loss.item())

            return result

    def evaluate_classification(
        self,
        sentences: List[Sentence],
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
    ) -> Result:

        # use scikit-learn to evaluate
        y_true = []
        y_pred = []

        with torch.no_grad():
            eval_loss = torch.zeros(1, device=flair.device)

            lines: List[str] = []
            # we iterate over each sentence, instead of batches
            for sentence in sentences:

                if len(sentence) < 2:  # we need at least 2 words per sentence
                    continue

                scores, loss = self._forward_scores_and_loss(sentence, return_loss=True)

                # get single labels from scores
                predictions = [self._get_single_label(s) for s in scores]

                # gold labels
                true_values_for_sentence = []
                numberOfPairs = 0
                numberOfWords = len(sentence)
                lines.append(sentence.to_tokenized_string() + "\n")
                for i in range(numberOfWords):
                    for j in range(i + 1, min(i + self.max_distance + 2, numberOfWords)):
                        true_values_for_sentence.append(j - i - 1)

                        # for output text file
                        eval_line = "({},{})\t{}\t{}\n".format(i, j, j - i - 1, predictions[numberOfPairs])
                        lines.append(eval_line)

                        numberOfPairs += 1

                eval_loss += loss / numberOfPairs  # add average loss of word pairs

                for prediction_for_sentence, true_value_for_sentence in zip(predictions, true_values_for_sentence):
                    # hot one vector of true value
                    y_true_instance = np.zeros(self.max_distance + 1, dtype=int)
                    y_true_instance[true_value_for_sentence] = 1
                    y_true.append(y_true_instance.tolist())

                    # hot one vector of predicted value
                    y_pred_instance = np.zeros(self.max_distance + 1, dtype=int)
                    y_pred_instance[prediction_for_sentence] = 1
                    y_pred.append(y_pred_instance.tolist())

                # speichert embeddings, falls embedding_storage!= 'None'
                store_embeddings([sentence], embedding_storage_mode)

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make "classification report"
            target_names = []  # liste aller labels, ins unserem Fall
            for i in range(self.max_distance + 1):
                target_names.append(str(i))
            classification_report = metrics.classification_report(
                y_true, y_pred, digits=4, target_names=target_names, zero_division=0
            )

            # get scores
            micro_f_score = round(
                metrics.fbeta_score(y_true, y_pred, beta=self.beta, average="micro", zero_division=0),
                4,
            )
            accuracy_score = round(metrics.accuracy_score(y_true, y_pred), 4)
            macro_f_score = round(
                metrics.fbeta_score(y_true, y_pred, beta=self.beta, average="macro", zero_division=0),
                4,
            )
            # precision_score = round(metrics.precision_score(y_true, y_pred, average='macro', zero_division=0), 4)
            # recall_score = round(metrics.recall_score(y_true, y_pred, average='macro', zero_division=0), 4)

            detailed_result = (
                "\nResults:"
                f"\n- F-score (micro) {micro_f_score}"
                f"\n- F-score (macro) {macro_f_score}"
                f"\n- Accuracy {accuracy_score}"
                "\n\nBy class:\n" + classification_report
            )

            # line for log file
            log_header = "ACCURACY"
            log_line = f"\t{accuracy_score}"
            eval_loss /= len(sentences)

            result = Result(
                main_score=micro_f_score,
                log_line=log_line,
                log_header=log_header,
                detailed_results=detailed_result,
                loss=eval_loss.item(),
            )

            return result

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning("Ignore {} sentence(s) with no tokens.".format(len(sentences) - len(filtered_sentences)))
        return filtered_sentences

    def _obtain_labels(self, scores: List[List[float]], predict_prob: bool = False) -> List[List[Label]]:
        """
        Predicts the labels of sentences.
        :param scores: the prediction scores from the model
        :return: list of predicted labels
        """

        if predict_prob:
            return [self._predict_label_prob(s) for s in scores]

        return [self._get_single_label(s) for s in scores]

    def _get_single_label(self, label_scores):  # -> List[Label]:
        softmax = torch.nn.functional.softmax(label_scores, dim=0)
        conf, idx = torch.max(softmax, 0)

        return idx.item()

    def _predict_label_prob(self, label_scores) -> List[Label]:
        softmax = torch.nn.functional.softmax(label_scores, dim=0)
        label_probs = []
        for idx, conf in enumerate(softmax):
            label_probs.append(Label(str(idx), conf.item()))
        return label_probs

    def __str__(self):
        return (
            super(flair.nn.Model, self).__str__().rstrip(")")
            + f"  (beta): {self.beta}\n"
            + f"  (loss_max_weight): {self.loss_max_weight}\n"
            + f"  (max_distance) {self.max_distance}\n)"
        )
