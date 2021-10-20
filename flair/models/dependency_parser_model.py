import enum
from pathlib import Path
from typing import List, Union, Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn

from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence
import sklearn
import numpy as np

import flair.nn
from flair.data import Dictionary, Sentence, Token, Label, DataPoint
from flair.datasets import DataLoader, SentenceDataset
from flair.embeddings import TokenEmbeddings
from flair.training_utils import Result, store_embeddings
from flair.nn.dropout import LockedDropout
from flair.visual.tree_printer import tree_printer

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DependencyParser(flair.nn.Model):
    def __init__(
            self,
            token_embeddings: TokenEmbeddings,
            relations_dictionary: Dictionary,
            lstm_hidden_size: int = 400,
            mlp_arc_units: int = 500,
            mlp_rel_units: int = 100,
            lstm_layers: int = 3,
            mlp_dropout: float = 0.1,
            lstm_dropout: float = 0.2,
        ):
        """
        Initializes a DependecyParser
        The model is based on biaffine dependency parser :cite: "Dozat T. & Manning C. Deep biaffine attention for neural dependency parsing."
        
        :param token_embeddings: word embeddings used in model
        :param relations_dictionary: dictionary of relations tags
        :param lstm_hidden_size: size of LSTM hidden state
        :param mlp_arc_units: size of MLP for arc 
        :param mlp_rel_units: size of MLP for dependency relations
        :param lstm_layers: number of LSTM layers
        :param mlp_dropout: The dropout probability of MLP layers
        :param lstm_dropout: dropout probability in LSTM 
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
        self.tag_type = 'dependency'
        self.lstm_input_dim: int = self.token_embeddings.embedding_length
        if self.relations_dictionary:
            self.embedding2nn = torch.nn.Linear(self.lstm_input_dim,
                                                self.lstm_input_dim)

        self.lstm = BiLSTM(input_size=self.lstm_input_dim,
                           hidden_size=self.lstm_hidden_size,
                           num_layers=self.lstm_layers,
                           dropout=self.lstm_dropout)

        self.mlp_arc_h = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.mlp_arc_units,
                             dropout=self.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.mlp_arc_units,
                             dropout=self.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.mlp_rel_units,
                             dropout=self.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.mlp_rel_units,
                             dropout=self.mlp_dropout)

        self.arc_attn = Biaffine(n_in=self.mlp_arc_units,
                                 bias_x=True,
                                 bias_y=False)

        self.rel_attn = Biaffine(n_in=self.mlp_rel_units,
                                 n_out=len(relations_dictionary),
                                 bias_x=True,
                                 bias_y=True)

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.to(flair.device)

    def forward(self, sentences: List[Sentence]):
        self.token_embeddings.embed(sentences)
        batch_size = len(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        seq_len: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(self.token_embeddings.embedding_length * seq_len,
                                                dtype=torch.float,
                                                device=flair.device)

        # embed sentences
        all_embs = list()
        for sentence in sentences:
            all_embs += [emb for token in sentence for emb in token.get_each_embedding()]
            nb_padding_tokens = seq_len - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[:self.token_embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view([batch_size, seq_len,
                                                    self.token_embeddings.embedding_length,])

        x = pack_padded_sequence(sentence_tensor, lengths, True, False)

        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)

        # apply MLPs for arc and relations to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get scores from the biaffine attentions
        # [batch_size, seq_len, seq_len]
        score_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        score_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return score_arc, score_rel
    

    def forward_loss(self, data_points: List[Sentence]) -> torch.tensor:
        
        score_arc, score_rel = self.forward(data_points)
        loss_arc, loss_rel = self._calculate_loss(score_arc, score_rel, data_points)
        main_loss = loss_arc + loss_rel

        return main_loss


    def _calculate_loss(self, score_arc: torch.tensor,
                        score_relation: torch.tensor,
                        data_points: List[Sentence]) -> Tuple[float, float]:

        arc_loss = 0.0
        rel_loss = 0.0

        for sen_id, sen in enumerate(data_points):

            arc_labels = [token.head_id - 1 if token.head_id != 0 else token.idx - 1 
                          for token in sen.tokens]
            arc_labels = torch.tensor(arc_labels, dtype=torch.int64, device=flair.device)
            arc_loss += self.loss_function(score_arc[sen_id], arc_labels)

            rel_labels = [self.relations_dictionary.get_idx_for_item(token.get_tag('dependency').value)
                          for token in sen.tokens]
            rel_labels = torch.tensor(rel_labels, dtype=torch.int64, device=flair.device)
            score_relation = score_relation[sen_id][torch.arange(len(arc_labels)), arc_labels]
            rel_loss += self.loss_function(score_relation, rel_labels)

        return arc_loss, rel_loss
    
    def predict(self,
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

        if not sentences:
            return sentences
        sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences,
                                 batch_size=mini_batch_size,
                                 num_workers=num_workers)

        for batch in data_loader:
            with torch.no_grad():
                score_arc, score_rel = self.forward(batch)
                arc_prediction, relation_prediction = self._obtain_labels_(score_arc, score_rel)

            for sentnce_index, (sentence, sent_tags, sent_arcs) in enumerate(zip(batch, relation_prediction, arc_prediction)):
                for token_index, (token, tag, head_id) in enumerate(zip(sentence.tokens, sent_tags, sent_arcs)):
                    token.add_tag(self.tag_type,
                                  tag,
                                  score_rel[sentnce_index][token_index])
                    
                    token.head_id = int(head_id)

                if print_tree:
                    tree_printer(sentence, self.tag_type)
                    print("-" * 50)
            store_embeddings(batch, storage_mode=embedding_storage_mode)
    
    def evaluate(
            self,
            data_points: Union[List[DataPoint], Dataset],
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
            main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        ) -> Result:
        
        if not isinstance(data_points, Dataset):
            data_points = SentenceDataset(data_points)
        data_loader = DataLoader(data_points,
                                 batch_size=mini_batch_size,
                                 num_workers=num_workers)

        lines: List[str] = []

        eval_loss_arc = 0
        eval_loss_rel = 0

        y_true = []
        y_pred = []

        parsing_metric = ParsingMetric()

        for batch in data_loader:

            with torch.no_grad():
                score_arc, score_rel = self.forward(batch)
                loss_arc, loss_rel = self._calculate_loss(score_arc, score_rel, batch)
                arc_prediction, relation_prediction = self._obtain_labels_(score_arc, score_rel)
                
            parsing_metric(arc_prediction, relation_prediction, batch)
            
            eval_loss_arc += loss_arc
            eval_loss_rel += loss_rel

            for (sentence, arcs, sent_tags) in zip(batch, arc_prediction, relation_prediction):
                for (token, arc, tag) in zip(sentence.tokens, arcs, sent_tags):
                    token: Token = token
                    token.add_tag_label("predicted", Label(tag))
                    token.add_tag_label("predicted_head_id", Label(str(arc)))

                    # append both to file for evaluation
                    eval_line = "{} {} {} {} {}\n".format(token.text,
                                                          token.get_tag('dependency').value,
                                                          str(token.head_id),
                                                          tag,
                                                          str(arc))
                    lines.append(eval_line)
                lines.append("\n")

            for sentence in batch:

                gold_tags = [token.get_tag('dependency').value for token in sentence.tokens]
                predicted_tags = [tag.tag for tag in sentence.get_spans("predicted")]

                y_pred += [self.relations_dictionary.get_idx_for_item(tag)
                           for tag in predicted_tags]
                y_true += [self.relations_dictionary.get_idx_for_item(tag)
                           for tag in gold_tags]

            store_embeddings(batch, embedding_storage_mode)

        eval_loss_arc /= len(data_loader)
        eval_loss_rel /= len(data_loader)

        if out_path is not None:
            with open(out_path, "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        classification_report_dict = sklearn.metrics.classification_report(y_true,
                                                                           y_pred,
                                                                           target_names=self.relations_dictionary.idx2item,
                                                                           zero_division=0,
                                                                           output_dict=True,
                                                                           labels=range(len(self.relations_dictionary)))

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
            loss=eval_loss_rel
        )
        return result, eval_loss_rel

    def _obtain_labels_(self, score_arc: torch.tensor, score_rel: torch.tensor) -> Tuple[List[List[int]],
                                                                                         List[List[str]]]:
        arc_prediction: torch.tensor = score_arc.argmax(-1)
        relation_prediction: torch.tensor = score_rel.argmax(-1)
        relation_prediction = relation_prediction.gather(-1, arc_prediction.unsqueeze(-1)).squeeze(-1)

        arc_prediction = [[arc+1 if token_index != arc else 0 for token_index, arc in enumerate(batch)]
                          for batch in arc_prediction]
        relation_prediction = [[self.relations_dictionary.get_item_for_index(rel_tag_idx)
                                for rel_tag_idx in batch] for batch in relation_prediction]

        return arc_prediction, relation_prediction
    
    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "token_embeddings": self.token_embeddings,
            "lstm_hidden_size": self.lstm_hidden_size,
            "relations_dictionary": self.relations_dictionary,
            "mlp_arc_units": self.mlp_arc_units,
            "mlp_rel_units": self.mlp_rel_units,
            "lstm_layers": self.lstm_layers,
            "lstm_dropout": self.lstm_dropout,
            "mlp_dropout": self.mlp_dropout,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        model = DependencyParser(
            token_embeddings=state["token_embeddings"],
            relations_dictionary=state["relations_dictionary"],
            lstm_hidden_size=state["lstm_hidden_size"],
            mlp_arc_units=state["mlp_arc_units"],
            mlp_rel_units=state["mlp_rel_units"],
            lstm_layers=state["lstm_layers"],
            mlp_dropout=state["mlp_dropout"],
            lstm_dropout=state["lstm_dropout"],
        )
        model.load_state_dict(state["state_dict"])
        return model
