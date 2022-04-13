import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import sklearn
import torch
import torch.nn
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset

import flair.nn
from flair.data import DataPoint, Dictionary, Sentence
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import TokenEmbeddings
from flair.nn.dropout import LockedDropout, WordDropout
from flair.training_utils import Result, log_line, store_embeddings
from flair.visual.tree_printer import tree_printer

log = logging.getLogger("flair")


class DependencyParser(flair.nn.Model):
    def __init__(
        self,
        token_embeddings: TokenEmbeddings,
        relations_dictionary: Dictionary,
        tag_type: str = "dependency",
        use_rnn: Union[bool, str] = True,
        lstm_hidden_size: int = 400,
        mlp_arc_units: int = 500,
        mlp_rel_units: int = 100,
        lstm_layers: int = 3,
        mlp_dropout: float = 0.33,
        lstm_dropout: float = 0.33,
        word_dropout: float = 0.05,
    ):
        """
        Initializes a DependencyParser
        The model is based on biaffine dependency parser :cite: "Dozat T. & Manning C. Deep biaffine attention for neural dependency parsing."

        :param token_embeddings: word embeddings used in model
        :param relations_dictionary: dictionary of relations tags
        :param string identifier for tag type
        :param lstm_hidden_size: size of LSTM hidden state
        :param mlp_arc_units: size of MLP for arc
        :param mlp_rel_units: size of MLP for dependency relations
        :param lstm_layers: number of LSTM layers
        :param mlp_dropout: The dropout probability of MLP layers
        :param lstm_dropout: dropout probability in LSTM
        :param word_dropout: word dropout probability
        """

        super(DependencyParser, self).__init__()

        self.token_embeddings = token_embeddings
        self.relations_dictionary: Dictionary = relations_dictionary
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_arc_units = mlp_arc_units
        self.mlp_rel_units = mlp_rel_units
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.mlp_dropout = mlp_dropout

        self.use_word_dropout: bool = word_dropout > 0
        if self.use_word_dropout:
            self.word_dropout = WordDropout(dropout_rate=word_dropout)

        self.tag_type = tag_type

        self.use_rnn = True

        # if there is no RNN
        if not use_rnn:

            self.use_rnn = False
            mlp_input_dim = self.token_embeddings.embedding_length

        else:

            self.lstm_input_dim: int = self.token_embeddings.embedding_length
            mlp_input_dim = self.lstm_hidden_size * 2

            if use_rnn == "Variational":
                self.lstm: torch.nn.Module = BiLSTM(
                    input_size=self.lstm_input_dim,
                    hidden_size=self.lstm_hidden_size,
                    num_layers=self.lstm_layers,
                    dropout=self.lstm_dropout,
                )
            else:
                self.lstm = torch.nn.LSTM(
                    self.lstm_input_dim,
                    self.lstm_hidden_size,
                    num_layers=self.lstm_layers,
                    dropout=lstm_dropout,
                    bidirectional=True,
                    batch_first=True,
                )

        self.mlp_arc_h = MLP(n_in=mlp_input_dim, n_hidden=self.mlp_arc_units, dropout=self.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=mlp_input_dim, n_hidden=self.mlp_arc_units, dropout=self.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=mlp_input_dim, n_hidden=self.mlp_rel_units, dropout=self.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=mlp_input_dim, n_hidden=self.mlp_rel_units, dropout=self.mlp_dropout)

        self.arc_attn = Biaffine(n_in=self.mlp_arc_units, bias_x=True, bias_y=False)

        self.rel_attn = Biaffine(n_in=self.mlp_rel_units, n_out=len(relations_dictionary), bias_x=True, bias_y=True)

        self.to(flair.device)

    def forward(self, sentences: List[Sentence]):
        self.token_embeddings.embed(sentences)
        batch_size = len(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        seq_len: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.token_embeddings.embedding_length * seq_len,
            dtype=torch.float,
            device=flair.device,
        )

        # embed sentences
        all_embs = list()
        for sentence in sentences:
            all_embs += [emb for token in sentence for emb in token.get_each_embedding()]
            nb_padding_tokens = seq_len - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[: self.token_embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                batch_size,
                seq_len,
                self.token_embeddings.embedding_length,
            ]
        )

        # Main model implementation drops words and tags (independently), instead, we use word dropout!
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        if self.use_rnn:
            sentence_sequence = pack_padded_sequence(sentence_tensor, torch.IntTensor(lengths), True, False)

            sentence_sequence, _ = self.lstm(sentence_sequence)
            sentence_tensor, _ = pad_packed_sequence(sentence_sequence, True, total_length=seq_len)

        # apply MLPs for arc and relations to the BiLSTM output states
        arc_h = self.mlp_arc_h(sentence_tensor)
        arc_d = self.mlp_arc_d(sentence_tensor)
        rel_h = self.mlp_rel_h(sentence_tensor)
        rel_d = self.mlp_rel_d(sentence_tensor)

        # get scores from the biaffine attentions
        # [batch_size, seq_len, seq_len]
        score_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        score_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return score_arc, score_rel

    def forward_loss(self, data_points: List[Sentence]) -> torch.Tensor:

        score_arc, score_rel = self.forward(data_points)
        loss_arc, loss_rel = self._calculate_loss(score_arc, score_rel, data_points)
        main_loss = loss_arc + loss_rel

        return main_loss

    def _calculate_loss(
        self,
        score_arc: torch.Tensor,
        score_relation: torch.Tensor,
        data_points: List[Sentence],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        lengths: List[int] = [len(sentence.tokens) for sentence in data_points]

        arc_loss = torch.zeros(1, device=flair.device)
        rel_loss = torch.zeros(1, device=flair.device)

        for sen_id, sen in enumerate(data_points):
            sen_len = lengths[sen_id]

            arc_labels_list = [
                token.head_id - 1 if token.head_id != 0 and token.head_id is not None else (token.idx or 0) - 1
                for token in sen.tokens
            ]
            arc_labels = torch.tensor(arc_labels_list, dtype=torch.int64, device=flair.device)
            arc_loss += torch.nn.functional.cross_entropy(score_arc[sen_id][:sen_len], arc_labels)

            rel_labels_list = [
                self.relations_dictionary.get_idx_for_item(token.get_label(self.tag_type).value) for token in sen.tokens
            ]

            rel_labels = torch.tensor(rel_labels_list, dtype=torch.int64, device=flair.device)
            score_rel = score_relation[sen_id][torch.arange(len(arc_labels)), arc_labels]
            rel_loss += torch.nn.functional.cross_entropy(score_rel, rel_labels)

        return arc_loss, rel_loss

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 32,
        num_workers: int = 8,
        print_tree: bool = False,
        embedding_storage_mode="none",
    ) -> None:
        """
        Predict arcs and tags for Dependency Parser task
        :param sentences: a Sentence or a List of Sentence
        :param mini_batch_size: mini batch size to use
        :param print_tree: set to True to print dependency parser of sentence as tree shape
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
        sentence_dataset = FlairDatapointDataset(sentences)
        data_loader = DataLoader(sentence_dataset, batch_size=mini_batch_size, num_workers=num_workers)

        for batch in data_loader:
            with torch.no_grad():
                score_arc, score_rel = self.forward(batch)
                arc_prediction, relation_prediction = self._obtain_labels_(score_arc, score_rel)

            for sentnce_index, (sentence, sent_tags, sent_arcs) in enumerate(
                zip(batch, relation_prediction, arc_prediction)
            ):

                for token_index, (token, tag, head_id) in enumerate(zip(sentence.tokens, sent_tags, sent_arcs)):
                    token.add_tag(self.tag_type, tag, score_rel[sentnce_index][token_index])

                    token.head_id = int(head_id)

                if print_tree:
                    tree_printer(sentence, self.tag_type)
                    log_line(log)
            store_embeddings(batch, storage_mode=embedding_storage_mode)

    def evaluate(
        self,
        data_points: Union[List[DataPoint], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        **kwargs,
    ) -> Result:

        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)
        data_loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=num_workers)

        lines: List[str] = ["token gold_tag gold_arc predicted_tag predicted_arc\n"]

        average_over = 0
        eval_loss_arc = 0.0
        eval_loss_rel = 0.0

        y_true = []
        y_pred = []

        parsing_metric = ParsingMetric()

        for batch in data_loader:
            average_over += 1
            with torch.no_grad():
                score_arc, score_rel = self.forward(batch)
                loss_arc, loss_rel = self._calculate_loss(score_arc, score_rel, batch)
                arc_prediction, relation_prediction = self._obtain_labels_(score_arc, score_rel)

            parsing_metric(arc_prediction, relation_prediction, batch, gold_label_type)

            eval_loss_arc += loss_arc.item()
            eval_loss_rel += loss_rel.item()

            for (sentence, arcs, sent_tags) in zip(batch, arc_prediction, relation_prediction):
                for (token, arc, tag) in zip(sentence.tokens, arcs, sent_tags):
                    token.add_label("predicted", value=tag)
                    token.add_label("predicted_head_id", value=str(int(arc)))

                    # append both to file for evaluation
                    eval_line = "{} {} {} {} {}\n".format(
                        token.text,
                        token.get_label(gold_label_type).value,
                        str(token.head_id),
                        tag,
                        str(int(arc)),
                    )
                    lines.append(eval_line)
                lines.append("\n")

            for sentence in batch:
                gold_tags = [token.get_label(gold_label_type).value for token in sentence]
                predicted_tags = [token.get_label("predicted").value for token in sentence]

                y_pred += [self.relations_dictionary.get_idx_for_item(tag) for tag in predicted_tags]
                y_true += [self.relations_dictionary.get_idx_for_item(tag) for tag in gold_tags]

            store_embeddings(batch, embedding_storage_mode)

        eval_loss_arc /= average_over
        eval_loss_rel /= average_over

        if out_path is not None:
            with open(out_path, "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        classification_report_dict = sklearn.metrics.classification_report(
            y_true,
            y_pred,
            target_names=self.relations_dictionary.idx2item,
            zero_division=0,
            output_dict=True,
            labels=range(len(self.relations_dictionary)),
        )

        accuracy_score = round(sklearn.metrics.accuracy_score(y_true, y_pred), 4)

        precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
        recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
        micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
        macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

        main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        detailed_result = (
            f"\nUAS : {parsing_metric.get_uas():.4f} - LAS : {parsing_metric.get_las():.4f}"
            f"\neval loss rel : {eval_loss_rel:.4f} - eval loss arc : {eval_loss_arc:.4f}"
            f"\nF-Score: micro : {micro_f_score} - macro : {macro_f_score}"
            f"\n Accuracy: {accuracy_score} - Precision {precision_score} - Recall {recall_score}"
        )
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        result = Result(
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=(eval_loss_rel + eval_loss_arc),
        )
        return result

    def _obtain_labels_(
        self, score_arc: torch.Tensor, score_rel: torch.Tensor
    ) -> Tuple[List[List[int]], List[List[str]]]:

        arc_prediction: torch.Tensor = score_arc.argmax(-1)
        relation_prediction: torch.Tensor = score_rel.argmax(-1)
        relation_prediction = relation_prediction.gather(-1, arc_prediction.unsqueeze(-1)).squeeze(-1)

        decoded_arc_prediction = [
            [arc + 1 if token_index != arc else 0 for token_index, arc in enumerate(batch)] for batch in arc_prediction
        ]
        decoded_relation_prediction = [
            [self.relations_dictionary.get_item_for_index(rel_tag_idx) for rel_tag_idx in batch]
            for batch in relation_prediction
        ]

        return decoded_arc_prediction, decoded_relation_prediction

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "token_embeddings": self.token_embeddings,
            "use_rnn": self.use_rnn,
            "lstm_hidden_size": self.lstm_hidden_size,
            "relations_dictionary": self.relations_dictionary,
            "mlp_arc_units": self.mlp_arc_units,
            "mlp_rel_units": self.mlp_rel_units,
            "lstm_layers": self.lstm_layers,
            "lstm_dropout": self.lstm_dropout,
            "mlp_dropout": self.mlp_dropout,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            token_embeddings=state["token_embeddings"],
            relations_dictionary=state["relations_dictionary"],
            use_rnn=state["use_rnn"],
            lstm_hidden_size=state["lstm_hidden_size"],
            mlp_arc_units=state["mlp_arc_units"],
            mlp_rel_units=state["mlp_rel_units"],
            lstm_layers=state["lstm_layers"],
            mlp_dropout=state["mlp_dropout"],
            lstm_dropout=state["lstm_dropout"],
            **kwargs,
        )

    @property
    def label_type(self):
        return self.tag_type


class BiLSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initializes a VariationalBiLSTM

        :param input_size: number of expected size in the input
        :param hidden_size: hidden state size
        :param num_layers: number of LSTM layers
        :param dropout: apply dropout on the output of eche LSTM cell as mentioned in paper
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.f_cells = torch.nn.ModuleList()
        self.b_cells = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.f_cells.append(torch.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            self.b_cells.append(torch.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            input_size = hidden_size * 2

        self.reset_parameters()

    def __repr__(self):
        st = "input:{} , hidden_size:{}, num_of_layers:{}, dropout_rate:{}".format(
            self.input_size, self.hidden_size, self.num_layers, self.dropout
        )
        return f"{self.__class__.__name__}({st})"

    def reset_parameters(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                torch.nn.init.orthogonal_(param)
            else:
                torch.nn.init.zeros_(param)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)

        return h, c

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = hx_0[0].new_empty(hx_0[0].shape).bernoulli_(1 - self.dropout) / (1 - self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size])) for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)

        return output, hx_n

    def forward(self, sequence, hx=None) -> Tuple[PackedSequence, torch.Tensor]:
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        if hx is None:
            ih = x.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, sequence.sorted_indices)
        h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
        c = c.view(self.num_layers, 2, batch_size, self.hidden_size)

        for i in range(self.num_layers):
            x = torch.split(x, batch_sizes)
            if self.training:
                mask = x[0].new_empty(x[0].shape).bernoulli_(1 - self.dropout) / (1 - self.dropout)
                x = [i * mask[: len(i)] for i in x]
            x_f, (h_f, c_f) = self.layer_forward(
                x=x,
                hx=(h[i, 0], c[i, 0]),
                cell=self.f_cells[i],
                batch_sizes=batch_sizes,
            )
            x_b, (h_b, c_b) = self.layer_forward(
                x=x,
                hx=(h[i, 1], c[i, 1]),
                cell=self.b_cells[i],
                batch_sizes=batch_sizes,
                reverse=True,
            )
            x = torch.cat((x_f, x_b), -1)
            h_n.append(torch.stack((h_f, h_b)))
            c_n.append(torch.stack((c_f, c_b)))
        x = PackedSequence(x, sequence.batch_sizes, sequence.sorted_indices, sequence.unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, sequence.unsorted_indices)

        return x, hx


class Biaffine(torch.nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        """
        :param n_in: size of input
        :param n_out: number of channels
        :param bias_x: set bias for x
        :param bias_x: set bias for y

        """
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = torch.nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        st = "n_in:{}, n_out:{}, bias_x:{}, bias_x:{}".format(self.n_in, self.n_out, self.bias_x, self.bias_y)
        return st

    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        s = s.squeeze(1)

        return s


class MLP(torch.nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        dropout: float = 0.0,
    ):
        super(MLP, self).__init__()

        self.linear = torch.nn.Linear(n_in, n_hidden)
        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.dropout = LockedDropout(dropout_rate=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class ParsingMetric:
    def __init__(self, epsilon=1e-8):

        self.eps = epsilon
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __call__(
        self,
        arc_prediction: List[List[int]],
        relation_prediction: List[List[str]],
        sentences: List[Sentence],
        tag_type: str,
    ):

        for batch_indx, batch in enumerate(sentences):
            self.total += len(batch.tokens)
            for token_indx, token in enumerate(batch.tokens):

                if arc_prediction[batch_indx][token_indx] == token.head_id:
                    self.correct_arcs += 1

                    # if head AND deprel correct, augment correct_rels score
                    if relation_prediction[batch_indx][token_indx] == token.get_label(tag_type).value:
                        self.correct_rels += 1

    def get_las(self) -> float:
        return self.correct_rels / (self.total + self.eps)

    def get_uas(self) -> float:
        return self.correct_arcs / (self.total + self.eps)
