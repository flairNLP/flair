import logging

from pathlib import Path
from typing import List, Union, Optional

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Sentence, Label
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings
from flair.training_utils import Metric, Result, store_embeddings

log = logging.getLogger("flair")


class SimpleSequenceTagger(flair.nn.Model):
    """
    This class is a simple version of the SequenceTagger class.
    The purpose of this class is to demonstrate the basic hierarchy of a
    sequence tagger (this could be helpful for new developers).
    It only uses the given embeddings and maps them with a linear layer to
    the tag_dictionary dimension.
    Thus, this class misses following functionalities from the SequenceTagger:
    - CRF,
    - RNN,
    - Reprojection.
    As a result, only poor results can be expected.
    """
    def __init__(
            self,
            embeddings: TokenEmbeddings,
            tag_dictionary: Dictionary,
            tag_type: str,
            beta: float = 1.0,
    ):
        """
        Initializes a SimpleSequenceTagger
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param beta: Parameter for F-beta score for evaluation and training annealing
        """

        super(SimpleSequenceTagger, self).__init__()

        # embeddings
        self.embeddings = embeddings

        # dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary
        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

        # linear layer
        self.linear = torch.nn.Linear(self.embeddings.embedding_length, len(tag_dictionary))

        # F-beta score
        self.beta = beta
     
        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    def forward_loss(
            self, data_points: Union[List[Sentence], Sentence], sort=True
    ) -> torch.tensor:
        features = self.forward(data_points)
        return self._calculate_loss(features, data_points)

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
                                label_name='predicted',
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

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "beta": self.beta,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        model = SimpleSequenceTagger(
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            beta=state["beta"],
        )
        model.load_state_dict(state["state_dict"])
        return model

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            mini_batch_size=32,
            all_tag_prob: bool = False,
            verbose: bool = False,
            label_name: Optional[str] = None,
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
        :param label_name: set this to change the name of the label type that is predicted
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if label_name is None:
            label_name = self.tag_type

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
                        token.add_tag_label(label_name, tag)

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (sentence, sent_all_tags) in zip(batch, all_tags):
                    for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                        token.add_tags_proba_dist(label_name, token_all_tags)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss / batch_no

    def forward(self, sentences: List[Sentence]):

        self.embeddings.embed(sentences)

        names = self.embeddings.get_names()

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding(names)
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.embeddings.embedding_length * nb_padding_tokens
                    ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )

        features = self.linear(sentence_tensor)

        return features

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
                sentence_feats, sentence_tags
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

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences

    def __str__(self):
        return super(flair.nn.Model, self).__str__().rstrip(')') + \
               f'  (beta): {self.beta}\n)'

    def _requires_span_F1_evaluation(self) -> bool:
        span_F1 = False
        for item in self.tag_dictionary.get_items():
            if item.startswith('B-'):
                span_F1 = True
        return span_F1

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
                                label_name='predicted',
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
