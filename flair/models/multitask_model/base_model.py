import logging
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
from tqdm import tqdm

import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data.dataset import Dataset

import flair.nn
from flair.data import Sentence, Dictionary, Label, Token
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings
from flair.training_utils import Result, store_embeddings
from flair.models.sequence_tagger_model import START_TAG, STOP_TAG

from .crf import CRF
from .viterbi import ViterbiLoss, ViterbiDecoder
from .utils import get_tags_tensor, init_stop_tag_embedding

log = logging.getLogger("flair")

class BaseModel(flair.nn.Model):
    """
    Basic multitask model.
    """

    def __init__(
         self,
         embeddings: TokenEmbeddings,
         tag_dictionary: Dictionary,
         tag_type: str,
         reproject_embeddings: bool = True,
         use_rnn: bool = True,
         rnn_type: str = "LSTM",
         hidden_size: int = 256,
         rnn_layers: int = 1,
         bidirectional: bool = True,
         use_crf: bool = True,
         dropout: float = 0.0,
         word_dropout: float = 0.05,
         locked_dropout: float = 0.5
    ):
        """
        Initializes a base multitask model instance
        :param embeddings: embeddings which are used
        :param tag_dictionary: Dictionary of tags of task
        :param tag_type: Type of tag which is going to be predicted
        :param reproject_embeddings: if True, adds trainable linear map on top of embedding layer. If False, no map.
        :param use_rnn: if True, adds a RNN layer to the model. If False, simple linear layer.
        :param rnn_type: specifies the RNN type to use. Use "RNN", "GRU" or "LSTM". Default is "LSTM".
        :param hidden_size: hidden size of the rnn layer.
        :param rnn_layers: number of layers to use for RNN.
        :param bidirectional: If True, RNN layer is bidirectional. If False, single direction.
        :param use_crf: If True, use Conditonal Random Field. If False, use Dense Softmax layer for prediction.
        :param use_lm: If True, use additional language model during training for multitask purpose.
        :param dropout: Includes standard dropout, if provided attribute value is > 0.0
        :param word_dropout: Includes word_dropout, if provided attribute value is > 0.0
        :param locked_dropout: Includes locked_dropout, if provided attribute value is > 0.0
        """

        super(BaseModel, self).__init__()

        # Embeddings and task specific attributes
        self.embeddings = embeddings
        self.stop_token_emb = init_stop_tag_embedding(embeddings.embedding_length)
        self.tag_dictionary = tag_dictionary
        self.tag_type = tag_type
        self.tagset_size: int = len(tag_dictionary)
        embedding_dim: int = embeddings.embedding_length
        if use_crf:
            self.tag_dictionary.add_item(START_TAG)
            self.tag_dictionary.add_item(STOP_TAG)

        # RNN specific attributes
        self.use_rnn = use_rnn
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        if reproject_embeddings or use_rnn:
            rnn_input_dim: int = embedding_dim

        # CRF and LM specific attributes
        self.use_crf = use_crf

        # Dropout specific attributes
        self.use_dropout = True if dropout > 0.0 else False
        self.use_word_dropout= True if word_dropout > 0.0 else False
        self.use_locked_dropout = True if locked_dropout > 0.0 else False

        # Model layers
        # Main task
        if self.use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        if self.use_word_dropout:
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        if self.use_locked_dropout:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        self.reproject_embeddings = reproject_embeddings
        if reproject_embeddings:
            if type(self.reproject_embeddings) == int:
                rnn_input_dim: int = self.reproject_embeddings
            self.embedding2nn = torch.nn.Linear(embedding_dim, rnn_input_dim)

        if use_rnn:
            self.rnn = self.RNN(rnn_type, rnn_layers,  hidden_size, bidirectional, rnn_input_dim)
        else:
            self.linear = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

        num_directions = 2 if bidirectional else 1
        if use_crf:
            self.crf = CRF(hidden_size * num_directions, self.tagset_size)
            self.viterbi_loss = ViterbiLoss(tag_dictionary)
        else:
            self.linear2tag = torch.nn.Linear(hidden_size * num_directions, len(tag_dictionary))
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        self.to(flair.device)

    @staticmethod
    def RNN(
        rnn_type: str,
        rnn_layers: int,
        hidden_size: int,
        bidirectional: bool,
        rnn_input_dim: int
    ):
        """
        Static wrapper function returning an RNN instance from PyTorch
        :param rnn_type: Type of RNN from torch.nn
        :param rnn_layers: number of layers to include
        :param hidden_size: hidden size of RNN cell
        :param bidirectional: If True, RNN cell is bidirectional
        :param rnn_input_dim: Input dimension to RNN cell
        """
        if rnn_type in ["LSTM", "GRU", "RNN"]:
            RNN = getattr(torch.nn, rnn_type)(
                    rnn_input_dim,
                    hidden_size,
                    num_layers=rnn_layers,
                    dropout=0.0 if rnn_layers == 1 else 0.5,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
        else:
            raise Exception(f"Unknown RNN type: {rnn_type}. Please use either LSTM, GRU or RNN.")

        return RNN

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """
        Forward loss function from abstract base class in flair
        :param sentences: list of sentences
        """
        features, lengths = self.forward(sentences)
        return self.loss(features, sentences, lengths)

    def forward(self, sentences: Union[List[Sentence], Sentence]):
        """
        Forward method of base multitask model
        :param sentences: list of sentences
        """

        # preparation of sentences and feed-forward part
        self.embeddings.embed(sentences)

        lengths = torch.LongTensor([len(sentence) + 1 for sentence in sentences])
        lengths = lengths.sort(dim=0, descending=True)

        tensor_list = list(map(lambda sent:
                               sent.get_sequence_tensor(add_stop_tag_embedding=True, stop_tag_embedding=self.stop_token_emb)
                               , sentences))
        sentence_tensor = pad_sequence(tensor_list, batch_first=True)
        # Sorted sentences in decreasing order
        sentence_tensor = sentence_tensor[lengths.indices]

        # feed-forward part
        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = pack_padded_sequence(sentence_tensor, list(lengths.values), batch_first=True)
            rnn_output, hidden = self.rnn(packed)
            sentence_tensor, output_lengths = pad_packed_sequence(rnn_output, batch_first=True)
        else:
            pass

        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.use_crf:
            features = self.crf(sentence_tensor)
        else:
            features = self.linear2tag(sentence_tensor)

        return features, lengths


    def loss(self, features: torch.Tensor, sentences: Union[List[Sentence], Sentence], lengths: tuple) -> torch.Tensor:
        """
        Loss function of multitask base model.
        :param features: Output features / CRF scores from feed-forward function
        :param sentences: list of sentences
        """

        # Preparation for loss function
        tags_tensor = get_tags_tensor(sentences, self.tag_dictionary, self.tag_type)
        # Sort tag tensor same order as features in decreasing order by length
        tags_tensor = tags_tensor[lengths.indices]

        if self.use_crf:
            loss = self.viterbi_loss(features, tags_tensor, lengths.values)
            #forward_score = self.loss_criterion(features, tags_tensor)
            #gold_score = self.loss_criterion.gold_score(features, tags_tensor, lengths)
            #loss = (forward_score - gold_score).mean()
        else:
            loss = self.cross_entropy_loss(features.permute(0,2,1), tags_tensor)

        return loss

    def evaluate(
            self,
            sentences: Union[List[Sentence], Dataset],
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
            wsd_evaluation: bool = False
    ) -> (Result, float):
        """
        Implementation of flair.nn.Model's interface, which evaluates current model
        :param sentences: List of sentences
        :param out_path: Path to store results in
        :param embedding_storage_mode: Whether to store embedding tensors on GPU or CPU
        :param mini_batch_size: batch size during prediction
        :param num_workers: amount of workers to use in torch DataLoader
        :param wsd_evaluation: True, if wsd evalution wanted
        """

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

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
                    if wsd_evaluation:
                        if gold_tag == 'O':
                            predicted_tag = 'O'
                        else:
                            predicted_tag = token.get_tag('predicted').value
                    else:
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
        if label_name == None:
            label_name = self.tag_type

        with torch.no_grad():
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

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

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
                    transitions=transitions,
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

    def _obtain_labels(
            self,
            feature: torch.Tensor,
            batch_sentences: List[Sentence],
            transitions: Optional[np.ndarray],
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
        if self.use_crf:
            feature = feature.numpy()
        else:
            for index, length in enumerate(lengths):
                feature[index, length:] = 0
            softmax_batch = F.softmax(feature, dim=2).cpu()
            scores_batch, prediction_batch = torch.max(softmax_batch, dim=2)
            feature = zip(softmax_batch, scores_batch, prediction_batch)

        for feats, length in zip(feature, lengths):
            if self.use_crf:
                confidences, tag_seq, scores = self._viterbi_decode(
                    feats=feats[:length],
                    transitions=transitions,
                    all_scores=get_all_tags,
                )
            else:
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