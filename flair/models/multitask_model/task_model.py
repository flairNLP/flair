from pathlib import Path
from typing import Union, List, Optional, Dict

import torch
import torch.nn
import torch.nn.functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import flair.nn
from flair.data import Sentence, Dictionary, Label
from flair.embeddings import TokenEmbeddings
from flair.training_utils import Metric, Result, store_embeddings, convert_labels_to_one_hot
from flair.models.sequence_tagger_model import START_TAG, STOP_TAG

from .crf import CRF
from .viterbi import ViterbiLoss, ViterbiDecoder
from .utils import init_stop_tag_embedding, get_tags_tensor

class SequenceTaggerTask(flair.nn.Model):

    def __init__(
            self,
            embeddings: TokenEmbeddings,
            tag_dictionary: Dictionary,
            tag_type: str,
            use_rnn: bool = True,
            rnn: Optional[torch.nn.Module] = None,
            rnn_type: str = "LSTM",
            hidden_size: int = 256,
            rnn_layers: int = 1,
            bidirectional: bool = True,
            use_crf: bool = True,
            reproject_embeddings: bool = True,
            dropout: float = 0.0,
            word_dropout: float = 0.0,
            locked_dropout: float = 0.5,
            beta: float = 1.0,
            loss_weights: Dict[str, float] = None,
    ):
        """
        Sequence Tagger class for predicting labels for single tokens. Can be parameterized by several attributes.
        In case of multitask learning, pass shared embeddings or shared rnn into respective attributes.
        :param embeddings: Embeddings to use during training and prediction
        :param tag_dictionary: Dictionary containing all tags from corpus which can be predicted
        :param tag_type: type of tag which is going to be predicted in case a corpus has multiple annotations
        :param use_rnn: If true, use a RNN, else Linear layer.
        :param rnn: (Optional) Takes a torch.nn.Module as parameter by which you can pass a shared RNN between
            different tasks.
        :param rnn_type: Specifies the RNN type to use, default is 'LSTM', can choose between 'GRU' and 'RNN' as well.
        :param hidden_size: Hidden size of RNN layer
        :param rnn_layers: number of RNN layers
        :param bidirectional: If True, RNN becomes bidirectional
        :param use_crf: If True, use a Conditional Random Field for prediction, else linear map to tag space.
        :param reproject_embeddings: If True, add a linear layer on top of embeddings, if you want to imitate
            fine tune non-trainable embeddings.
        :param dropout: If > 0, then use dropout.
        :param word_dropout: If > 0, then use word dropout.
        :param locked_dropout: If > 0, then use locked dropout.
        :param beta: Beta value for evaluation metric.
        :param loss_weights: Dictionary of weights for labels for the loss function
            (if any label's weight is unspecified it will default to 1.0)
        """
        super(SequenceTaggerTask, self).__init__()

        # ----- Multitask logging info -----
        self.name = f"{self._get_name()} - Task: {tag_type}"

        # ----- Embedding specific parameters -----
        self.embeddings = embeddings
        embedding_dim: int = embeddings.embedding_length
        self.stop_token_emb = init_stop_tag_embedding(embedding_dim)
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)
        self.tag_type = tag_type

        # ----- Evaluation metric parameters -----
        self.metric = Metric("Evaluation", beta=beta)
        self.beta = beta

        # ----- Initial loss weights parameters -----
        self.weight_dict = loss_weights
        self.loss_weights = self.init_loss_weights(loss_weights) if loss_weights else None

        # ----- RNN specific parameters -----
        self.use_rnn = use_rnn
        self.rnn_type = rnn_type if not rnn else rnn._get_name()
        self.hidden_size = hidden_size if not rnn else rnn.hidden_size
        self.rnn_layers = rnn_layers if not rnn else rnn.num_layers
        self.bidirectional = bidirectional if not rnn else rnn.bidirectional

        # ----- Conditional Random Field parameters -----
        self.use_crf = use_crf
        if use_crf:
            self.tag_dictionary.add_item(START_TAG)
            self.tag_dictionary.add_item(STOP_TAG)

        # ----- Dropout parameters -----
        self.use_dropout = True if dropout > 0.0 else False
        self.use_word_dropout = True if word_dropout > 0.0 else False
        self.use_locked_dropout = True if locked_dropout > 0.0 else False

        # ----- Model layers -----
        self.reproject_embeddings = reproject_embeddings
        if self.reproject_embeddings:
            self.embedding2nn = torch.nn.Linear(embedding_dim, embedding_dim)

        if self.use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        if self.use_word_dropout:
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        if self.use_locked_dropout:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        if use_rnn:
            # If Shared RNN provided, create one for model
            if not rnn:
                self.rnn = self.RNN(rnn_type, rnn_layers,  hidden_size, bidirectional, rnn_input_dim=embedding_dim)
            else:
                self.rnn = rnn
            num_directions = 2 if self.bidirectional else 1
            hidden_output_dim = self.rnn.hidden_size * num_directions
        else:
            self.linear = torch.nn.Linear(embedding_dim, embedding_dim)
            hidden_output_dim = embedding_dim

        if use_crf:
            self.crf = CRF(hidden_output_dim, self.tagset_size)
            self.viterbi_loss = ViterbiLoss(tag_dictionary)
            self.viterbi_decoder = ViterbiDecoder(tag_dictionary)
        else:
            self.linear2tag = torch.nn.Linear(hidden_output_dim, self.tagset_size)
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=self.loss_weights)

    def init_loss_weights(self, loss_weights) -> torch.Tensor:
        """
        Initialize loss weights.
        """
        n_classes = len(self.label_dictionary)
        weight_list = [1. for i in range(n_classes)]
        for i, tag in enumerate(self.label_dictionary.get_items()):
            if tag in loss_weights.keys():
                weight_list[i] = loss_weights[tag]
        return torch.FloatTensor(weight_list).to(flair.device)

    @staticmethod
    def RNN(
            rnn_type: str,
            rnn_layers: int,
            hidden_size: int,
            bidirectional: bool,
            rnn_input_dim: int
    ) -> torch.nn.Module:
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
        Forward loss function from abstract base class flair.nn.Model
        :param sentences: batch of sentences
        """
        features, lengths = self.forward(sentences)
        return self.loss(features, sentences, lengths)

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> (torch.Tensor, torch.Tensor):
        """
        Forward method of base multitask model
        :param sentences: list of sentences
        :return: features and lengths of each sentence in batch
        """
        self.embeddings.embed(sentences)

        # Get embedding for each sentence + append a stop token embedding to each sentence
        tensor_list = list(map(lambda sent: torch.cat((sent.get_sequence_tensor(), self.stop_token_emb.unsqueeze(0)), dim=0), sentences))
        sentence_tensor = pad_sequence(tensor_list, batch_first=True)

        # +1 since we've added a stop token embedding to each sentence
        lengths = torch.LongTensor([len(sentence) + 1 for sentence in sentences])
        lengths = lengths.sort(dim=0, descending=True)
        # sort tensor in decreasing order based on lengths of sentences in batch
        sentence_tensor = sentence_tensor[lengths.indices]

        # ----- Forward Propagation -----
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
            sentence_tensor = self.linear(sentence_tensor)

        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.use_crf:
            features = self.crf(sentence_tensor)
        else:
            features = self.linear2tag(sentence_tensor)

        return features, lengths

    def loss(self, features: torch.Tensor, sentences: Union[List[Sentence], Sentence], lengths) -> torch.Tensor:
        """
        Loss function of multitask base model.
        :param features: Output features / CRF scores from feed-forward function
        :param sentences: batch of sentences
        :param lengths: lenghts of sentences in batch to sort tag tensor accordingly
        """
        tags_tensor = get_tags_tensor(sentences, self.tag_dictionary, self.tag_type)
        tags_tensor = tags_tensor[lengths.indices]

        if self.use_crf:
            loss = self.viterbi_loss(features, tags_tensor, lengths.values)
        else:
            loss = self.cross_entropy_loss(features.permute(0,2,1), tags_tensor)

        return loss

    def evaluate(
        self,
        sentences: Union[List[Sentence], Sentence],
        out_path: Optional[Path] = None,
        embedding_storage_mode: str = "none",
    ) -> torch.Tensor:
        """
        flair.nn.Model interface implementation - evaluates the current model by predicting,
            calculating the respective metric and store the results.
        :param sentences: batch of sentences
        :param out_path: (Optional) output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
            freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :return: Returns a loss float value (Tensor) and stores a Result object as instance variable
        """
        with torch.no_grad():

            loss = self.predict(sentences,
                                embedding_storage_mode=embedding_storage_mode,
                                label_name='predicted',
                                return_loss=True)

            self.calculate_metric(sentences, out_path)

            self.store_result()

        return loss

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            label_name: Optional[str] = None,
            return_loss: bool = False,
            embedding_storage_mode: str ="none",
    ) -> Optional[torch.Tensor]:
        """
        Predicting tag sequence for current batch of sentences.
        :param sentences: batch of sentences
        :param label_name: which label should be predicted
        :param return_loss: If True, a loss float tensor is returned
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
            freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :return: Can return a loss float value (Tensor)
        """
        if label_name == None:
            label_name = self.tag_type

        features, lengths = self.forward(sentences)

        # features und lengths in der forward sortiert
        tags = self.viterbi_decoder.decode(features, lengths)

        # sorted sentences to match tags from decoder
        sentences = [sentences[i] for i in lengths.indices]

        # Add predicted labels to sentences
        for (sentence, sent_tags) in zip(sentences, tags):
            for (token, tag) in zip(sentence.tokens, sent_tags):
                token.add_tag_label(label_name, tag)

        # clearing token embeddings to save memory
        store_embeddings(sentences, storage_mode=embedding_storage_mode)

        if return_loss:
            return self.loss(features, sentences, lengths)

    def calculate_metric(self, sentences: Union[List[Sentence], Sentence], out_path: Union[str, Path] = None):
        """
        Calculates and stores a specific metric based on current predictions.
        :param sentences: batch of sentences with 'predicted' tags
        """
        # Some tagging tasks need span evaluation, i.e. named entity recognition.
        # since we need to handle [B-PER, I-PER] as on tag which needs to be predicted together to be correct.
        if self._requires_span_F1_evaluation():
            self._span_F1_evaluation(sentences, out_path)
        else:
            self._tag_F1_evaluation(sentences, out_path)

    def _requires_span_F1_evaluation(self) -> bool:
        """
        Check if we need to evaluate over spans of tags.
        :return: True if evaluate of span of tags
        """
        span_F1 = False
        for item in self.tag_dictionary.get_items():
            if item.startswith('B-'):
                span_F1 = True
        return span_F1

    def _span_F1_evaluation(self, sentences: Union[List[Sentence], Sentence], out_path: Union[str, Path] = None,):
        """
        Evaluates the predictions in each sentences of spans to token, i.e. for named
            entity recognition.
        :param sentences: batch of sentences
        """
        log_lines = []

        for sentence in sentences:

            gold_spans = sentence.get_spans(self.tag_type)
            gold_tags = [(span.tag, repr(span)) for span in gold_spans]

            predicted_spans = sentence.get_spans("predicted")
            predicted_tags = [(span.tag, repr(span)) for span in predicted_spans]

            for tag, prediction in predicted_tags:
                if (tag, prediction) in gold_tags:
                    self.metric.add_tp(tag)
                else:
                    self.metric.add_fp(tag)

            for tag, gold in gold_tags:
                if (tag, gold) not in predicted_tags:
                    self.metric.add_fn(tag)

            if out_path:
                for token in sentence:

                    gold_tag = 'O'
                    for span in gold_spans:
                        if token in span:
                            gold_tag = 'B-' + span.tag if token == span[0] else 'I-' + span.tag

                    predicted_tag = 'O'
                    for span in predicted_spans:
                        if token in span:
                            predicted_tag = 'B-' + span.tag if token == span[0] else 'I-' + span.tag

                    log_lines.append(f'{token.text} {gold_tag} {predicted_tag}\n')

            log_lines.append('\n')

        if out_path:
            with open(Path(out_path), "w", encoding="utf-8") as outfile:
                outfile.write("".join(log_lines))

    def _tag_F1_evaluation(self, sentences: Union[List[Sentence], Sentence], out_path: Union[str, Path] = None):
        """
        Evaluates the predictions in each sentences for single tags.
        :param sentences: batch of sentences
        """
        log_lines = []

        for sentence in sentences:

            for token in sentence:
                # add gold tag
                gold_tag = token.get_tag(self.tag_type).value
                predicted_tag = token.get_tag('predicted').value

                if gold_tag == predicted_tag:
                    self.metric.add_tp(predicted_tag)
                else:
                    self.metric.add_fp(predicted_tag)
                    self.metric.add_fn(gold_tag)

                log_lines.append(f'{token.text} {gold_tag} {predicted_tag}\n')

            log_lines.append('\n')

        if out_path:
            with open(Path(out_path), "w", encoding="utf-8") as outfile:
                outfile.write("".join(log_lines))

    def store_result(self):
        """
        Logging method which stores current results from metric
        in self.result which can be later used for logging.
        """
        detailed_result = (
            "\nResults:"
            f"\n- F1-score (micro) {self.metric.micro_avg_f_score():.4f}"
            f"\n- F1-score (macro) {self.metric.macro_avg_f_score():.4f}"
            '\n\nBy class:'
        )

        for class_name in self.metric.get_classes():
            detailed_result += (
                f"\n{class_name:<10} tp: {self.metric.get_tp(class_name)} - fp: {self.metric.get_fp(class_name)} - "
                f"fn: {self.metric.get_fn(class_name)} - precision: "
                f"{self.metric.precision(class_name):.4f} - recall: {self.metric.recall(class_name):.4f} - "
                f"f1-score: "
                f"{self.metric.f_score(class_name):.4f}"
            )

        self.result = Result(
            main_score=self.metric.micro_avg_f_score(),
            log_line=f"{self.metric.precision():.4f}\t{self.metric.recall():.4f}\t{self.metric.micro_avg_f_score():.4f}",
            log_header="PRECISION\tRECALL\tF1",
            detailed_results=detailed_result,
            multitask_id=self.name,
        )

    def _reset_eval_metrics(self):
        """
        Resets current metric and result, i.e. can be called after
        each evaluation batch of multitask model.
        """
        self.metric = Metric("Evaluation", beta=self.beta)
        self.result = None

    def _get_state_dict(self):
        """Returns the state dictionary for this model."""
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "rnn_type": self.rnn_type,
            "beta": self.beta,
            "reproject_embeddings": self.reproject_embeddings,
            "weight_dict": self.weight_dict
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        """Initialize the model from a state dictionary."""
        rnn_type = "LSTM" if "rnn_type" not in state.keys() else state["rnn_type"]
        use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
        use_word_dropout = 0.0 if "use_word_dropout" not in state.keys() else state["use_word_dropout"]
        use_locked_dropout = 0.0 if "use_locked_dropout" not in state.keys() else state["use_locked_dropout"]
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        reproject_embeddings = True if "reproject_embeddings" not in state.keys() else state["reproject_embeddings"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]

        model = SequenceTaggerTask(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            rnn_type=rnn_type,
            beta=beta,
            reproject_embeddings=reproject_embeddings,
            loss_weights=weights
        )
        model.load_state_dict(state["state_dict"])
        return model

class TextClassificationTask(flair.nn.Model):

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
        Text classification class for predicting labels for entire sentences. Can be parameterized by several attributes.
        In case of multitask learning, pass shared embeddings or shared rnn into respective embeddings class, which are
        then passed in to single TextClassificationTask instances.
        :param document_embeddings: Document Embeddings to use during training and prediction
        :param label_dictionary: Dictionary containing all labels from corpus
        :param label_type: type of label which is going to be predicted
        :param multi_label: If true, a sentence can have multiple labels. Do not mess up with multiclass prediction.
        :param multi_label_threshold: Confidence threshold when a label counts as predicted.
        :param loss_weights: Dictionary of weights for labels for the loss function
            (if any label's weight is unspecified it will default to 1.0)
        :param beta: Beta value for evaluation metric.
        """
        super(TextClassificationTask, self).__init__()

        # ----- Multitask logging info -----
        self.name = f"{self._get_name()} - Task: {label_type}"

        # ----- Embedding and label parameters -----
        self.document_embeddings: flair.embeddings.DocumentEmbeddings = document_embeddings
        self.label_dictionary: Dictionary = label_dictionary
        self.label_type = label_type

        if multi_label is not None:
            self.multi_label = multi_label
        else:
            self.multi_label = self.label_dictionary.multi_label
        self.multi_label_threshold = multi_label_threshold

        # ----- Evaluation metric parameters -----
        self.metric = Metric("Evaluation", beta=beta)
        self.beta = beta

        # ----- Initial loss weights parameters -----
        self.weight_dict = loss_weights
        self.loss_weights = self.init_loss_weights(loss_weights) if loss_weights else None

        # ----- Model layers -----
        self.decoder = torch.nn.Linear(self.document_embeddings.embedding_length, len(self.label_dictionary))
        torch.nn.init.xavier_uniform_(self.decoder.weight)

        if self.multi_label:
            self.loss_function = torch.nn.BCEWithLogitsLoss(weight=self.loss_weights)
        else:
            self.loss_function = torch.nn.CrossEntropyLoss(weight=self.loss_weights)

    def init_loss_weights(self, loss_weights) -> torch.Tensor:
        """
        Initialize loss weights.
        """
        n_classes = len(self.label_dictionary)
        weight_list = [1. for i in range(n_classes)]
        for i, tag in enumerate(self.label_dictionary.get_items()):
            if tag in loss_weights.keys():
                weight_list[i] = loss_weights[tag]
        return torch.FloatTensor(weight_list).to(flair.device)

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.tensor:
        """
        Forward loss implementation of flair.nn.Models interface.
        :param sentences: batch of sentences
        :return: float loss in tensor data type
        """
        scores = self.forward(sentences)
        return self.loss(scores, sentences)

    def forward(self, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """
        Forward propagation of text classification.
        :param sentences: batch of sentences
        :return: tensor containing the scores per label
        """
        self.document_embeddings.embed(sentences)

        embedding_names = self.document_embeddings.get_names()

        text_embedding_list = [sentence.get_embedding(embedding_names).unsqueeze(0) for sentence in sentences]
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)

        # ----- Forward Propagation -----
        label_scores = self.decoder(text_embedding_tensor)

        return label_scores

    def loss(self, scores: torch.Tensor, sentences: Union[List[Sentence], Sentence]) -> torch.Tensor:
        """
        Calculates the loss given scores per labels per token in sentence, and sentence batch.
        :param scores: Tensor containing scores per lablel per token per sentence
        :param sentences: annotated batch of sentences (predicted and true tags)
        :return: loss tensor
        """
        if self.multi_label:
            labels = self._labels_to_one_hot(sentences)
        else:
            labels = self._labels_to_indices(sentences)

        return self.loss_function(scores, labels)

    def _labels_to_one_hot(self, sentences: List[Sentence]) -> torch.Tensor:
        """
        convert sentences labels to one hot encoded tensors.
        :param sentences: batch of sentences
        :return: tensor with one-hot encoded labels
        """
        label_list = []
        for sentence in sentences:
            label_list.append([label.value for label in sentence.get_labels(self.label_type)])

        one_hot = convert_labels_to_one_hot(label_list, self.label_dictionary)
        one_hot = [torch.FloatTensor(l).unsqueeze(0) for l in one_hot]
        one_hot = torch.cat(one_hot, 0).to(flair.device)
        return one_hot

    def _labels_to_indices(self, sentences: List[Sentence]) -> torch.Tensor:
        """
        convert sentences labels indices tensors (indices from tag dictionary).
        :param sentences: batch of sentences
        :return: tensor with indices encoded labels
        """
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

    def evaluate(
            self,
            sentences: Union[List[Sentence], Sentence],
            out_path: Path = None,
            embedding_storage_mode: str = "none"
    )-> torch.Tensor:
        """
        flair.nn.Model interface implementation - evaluates the current model by predicting,
            calculating the respective metric and store the results.
        :param sentences: batch of sentences
        :param out_path: (Optional) output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
            freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :return: Returns a loss float value (Tensor) and stores a Result object as instance variable
        """
        with torch.no_grad():

            loss = self.predict(sentences,
                                embedding_storage_mode=embedding_storage_mode,
                                label_name='predicted',
                                return_loss=True)

            self.calculate_metric(sentences, out_path)

            self.store_result()

        return loss

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence],
            label_name: Optional[str] = None,
            embedding_storage_mode="none",
            return_loss=False,
    ):
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.
        :param sentences: batch of sentences
        :param label_name: label name to be predicted
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
            freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :param return_loss: If true, return loss
        """
        if label_name == None:
            label_name = self.label_type if self.label_type is not None else 'label'

        features = self.forward(sentences)

        loss = self.loss(features, sentences)

        predicted_labels = self._obtain_labels(features)

        for (sentence, labels) in zip(sentences, predicted_labels):
            for label in labels:
                if self.multi_label:
                    sentence.add_label(label_name, label.value, label.score)
                else:
                    sentence.set_label(label_name, label.value, label.score)

        store_embeddings(sentences, storage_mode=embedding_storage_mode)

        if return_loss:
            return loss

    def _obtain_labels(self, scores: torch.Tensor) -> List[List[Label]]:
        """
        Extract label class instances for each score.
        """
        if self.multi_label:
            return [self._get_multi_label(s) for s in scores]
        return [self._get_single_label(s) for s in scores]

    def _get_single_label(self, label_scores: torch.Tensor) -> List[Label]:
        """
        Max + Softmax over label scores, get respective id from label dictionary
        and return List of Labels.
        :param label_scores: scores per label per token per sentence
        """
        softmax = torch.nn.functional.softmax(label_scores, dim=0)
        conf, idx = torch.max(softmax, 0)
        label = self.label_dictionary.get_item_for_index(idx.item())
        return [Label(label, conf.item())]

    def _get_multi_label(self, label_scores: torch.Tensor) -> List[Label]:
        """
        Sigmoid over label scores, if score is higher then threshold,
        append Label to returned label list.
        :param label_scores: scores per label per token per sentence
        """
        labels = []
        sigmoid = torch.nn.Sigmoid()
        results = list(map(lambda scores: sigmoid(scores), label_scores))
        for idx, conf in enumerate(results):
            if conf > self.multi_label_threshold:
                label = self.label_dictionary.get_item_for_index(idx)
                labels.append(Label(label, conf.item()))
        return labels

    def calculate_metric(self, sentences: Union[List[Sentence], Sentence], out_path: Union[str, Path] = None):
        """
        Calculates and stores a specific metric based on current predictions.
        :param sentences: batch of sentences with 'predicted' tags
        """
        predicted_labels_batch = list(map(lambda sentence: sentence.get_labels('predicted'), sentences))
        list(map(lambda sentence: sentence.remove_labels('predicted'), sentences))
        gold_labels_batch = list(map(lambda sentence: sentence.get_labels(self.label_type), sentences))

        for (gold_labels, predicted_labels) in zip(gold_labels_batch, predicted_labels_batch):
            gold_labels = [label.value for label in gold_labels]
            predicted_labels = [label.value for label in predicted_labels]

            for prediction in predicted_labels:
                if prediction in gold_labels:
                    self.metric.add_tp(prediction)
                else:
                    self.metric.add_fp(prediction)

            for gold_label in gold_labels:
                if gold_label not in predicted_labels:
                    self.metric.add_fn(gold_label)

        if out_path:

            log_lines = []

            for sentence, prediction, gold in zip(sentences, predicted_labels_batch, gold_labels_batch):
                eval_line = "{}\t{}\t{}\n".format(sentence, gold, prediction)
                log_lines.append(eval_line)

            with open(out_path, "w", encoding="utf-8") as outfile:
                outfile.write("".join(log_lines))

    def store_result(self):
        """
        Logging method which stores current results from metric
        in self.result which can be later used for logging.
        """
        detailed_result = (
                "\nResults:"
                f"\n- F-score (micro) {self.metric.micro_avg_f_score():.4f}"
                f"\n- F-score (macro) {self.metric.macro_avg_f_score():.4f}"
                '\n\nBy class:'
        )

        for class_name in self.metric.get_classes():
            detailed_result += (
                f"\n{class_name:<10} tp: {self.metric.get_tp(class_name)} - fp: {self.metric.get_fp(class_name)} - "
                f"fn: {self.metric.get_fn(class_name)} - precision: "
                f"{self.metric.precision(class_name):.4f} - recall: {self.metric.recall(class_name):.4f} - "
                f"f1-score: "
                f"{self.metric.f_score(class_name):.4f}"
            )

        if not self.multi_label:
            log_header = "ACCURACY"
            log_line = f"\t{self.metric.accuracy():.4f}"
        else:
            log_header = "PRECISION\tRECALL\tF1\tACCURACY"
            log_line = f"{self.metric.precision()}\t" \
                       f"{self.metric.recall()}\t" \
                       f"{self.metric.macro_avg_f_score()}\t" \
                       f"{self.metric.accuracy()}"

        self.result = Result(
            main_score=self.metric.f_score(),
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            multitask_id=self.name
        )

    def _reset_eval_metrics(self):
        """
        Resets current metric and result, i.e. can be called after
        each evaluation batch of multitask model.
        """
        self.metric = Metric("Evaluation", beta=self.beta)
        self.result = None

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

        model = TextClassificationTask(
            document_embeddings=state["document_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=label_type,
            multi_label=state["multi_label"],
            beta=beta,
            loss_weights=weights,
        )

        model.load_state_dict(state["state_dict"])
        return model