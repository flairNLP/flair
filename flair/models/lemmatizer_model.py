import logging

from pathlib import Path
from typing import List, Union, Tuple, Optional

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

import flair.nn
import flair.embeddings
from flair.data import Sentence, Dictionary
from flair.training_utils import Result, store_embeddings
from flair.datasets import DataLoader, SentenceDataset


log = logging.getLogger("flair")

class Lemmatizer(flair.nn.Model):

    def __init__(self, token_embedding: flair.embeddings.TokenEmbeddings,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers_in_rnn: int = 1,
                 path_to_char_dict: str = None,
                 label_name = 'lemma'):
        """
        Initializes a Lemmatizer model
        The model consits of a decoder and an encoder. The decoder is a Token-Embedding from flair. The embedding of a token
        is handed as the initial hidden state to the decoder, which is a RNN-cell (torch.nn.GRU) that predicts the lemma of
        the given token one letter at a time.
        :param token_embedding: Embedding used to encode sentence
        :param input_size: Input size of the RNN. Each letter of a token is represented by a hot-one-vector over the given character
            dictionary. This vector is transformed to a input_size vector with a linear layer.
        :param hidden_size: size of the hidden state of the RNN. The initial embedding is transformed to a vector of size hidden_size
            with an additional linear layer
        :param num_layers_in_rnn: Number of stacked RNN cells
        :param path_to_char_dict: Path to character dictionary. If None, a standard dictionary will be used.
            Note that there are some rules to the dictionary. The first three indices must be reserved for the special characters
            <>, <S>, <E>, in that order. The characters represent a dummy symbol, start and end of a word, respectively.
        :param label_name: Name of the gold labels to use.
        """

        super().__init__()

        # encoder
        self.token_embedding = token_embedding
        self.hidden_size = hidden_size

        # decoder
        self.input_size = input_size
        self.rnn = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=num_layers_in_rnn)
        self.num_layers = num_layers_in_rnn

        self.loss = nn.CrossEntropyLoss()

        # character embedding
        self.path_to_char_dict = path_to_char_dict
        if path_to_char_dict is None:
            #self.char_dictionary: Dictionary = Dictionary.load('C:\\Users\\Marcel\\Desktop\\Arbeit\\Task\\Lemmatization\\char_dict\\dic')
            self.char_dictionary: Dictionary = Dictionary.load("common-chars-lemmatizer")
        else:
            self.char_dictionary: Dictionary = Dictionary.load_from_file(path_to_char_dict)
        # TODO: fct that creates char_dict from train_data?!

        self.alphabet_size = len(self.char_dictionary)

        # linear layers to transform vectors to and from alphabet_size
        self.character_embedding = nn.Embedding(self.alphabet_size, input_size)
        self.character_decoder = nn.Linear(self.hidden_size, self.alphabet_size)

        # linear layer to transform embedding vectors into hidden size
        self.emb_to_hidden = nn.Linear(token_embedding.embedding_length, hidden_size)

        self.label_name = label_name

        # Softmax for prediction
        self.softmax = nn.Softmax(dim=2)

        self.to(flair.device)

    def label_type(self):
        return self.label_name

    def labels_to_char_indices(self, labels: List[str], for_input = True):
        """
        For a given list of labels (lemmas) this function creates index vectors that represent the characters of the single words in the sentence.
        The "batch size" is given by the number of words in the list. Then each word is represented by sequence_length (maximum word length in the
        list) many indices representing characters in self.char_dict.
        """
        # sequence length of each word is equal to max length of a word in the sentence plus one (start character <S>)
        sequence_length = max(len(label) for label in labels) + 1
        tensor = torch.zeros(len(labels), sequence_length, dtype=torch.long).to(flair.device) # batch size is length of sentence
        for i in range(len(labels)):
            if for_input:
                tensor[i][0] = 1 # start character <S>
                for index, letter in enumerate(labels[i]):
                    try:
                        tensor[i][index + 1] = self.char_dictionary.get_idx_for_item(letter)
                    except:
                        print(f"Unknown character '{letter}'. Ignore corresponding sentence/batch.")
                        return None
            else:
                tensor[i][len(labels[i])] = 2 # end character <E> in the end
                for index, letter in enumerate(labels[i]):
                    try:
                        tensor[i][index] = self.char_dictionary.get_idx_for_item(letter)
                    except:
                        print(f"Unknown character '{letter}'. Ignore corresponding sentence/batch.")
                        return None
        return tensor
    
    def forward_pass(self, sentences: Union[List[Sentence], Sentence]):
        # pass (list of) sentence(s) through encoder-decoder

        if isinstance(sentences, Sentence):
            sentences = [sentences]

        # embedd sentences
        self.token_embedding.embed(sentences)

        # create list of all tokens of batch, this way we can hand over all sentences at once
        tokens = [token for sentence in sentences for token in sentence]
        
        # create inital hidden state tensor for batch (num_layers, batch_size, hidden_size)
        initial_hidden_states = self.emb_to_hidden(torch.stack(self.num_layers * [torch.stack([token.get_embedding() for token in tokens])]))

        #get labels, if no label provided take the word itself
        labels = [token.get_tag(label_type=self.label_name).value  if token.get_tag(label_type=self.label_name).value else token.text for token in tokens]
       
        # get char indices for labels of sentence
        # (batch_size, max_sequence_length) batch_size = #words in sentence, max_sequence_length = length of longet label of sentence
        input_indices = self.labels_to_char_indices(labels, for_input=True)

        if input_indices == None: # unknown letter in sentence
            return None, None

        # get char embeddings
        # (batch_size,max_sequence_length,input_size), i.e. replaces char indices with vectors of length input_size
        input_tensor = self.character_embedding(input_indices)
        
        # pass batch through rnn
        output, hn = self.rnn(input_tensor, initial_hidden_states)
    
        # transform output to vectors of size self.alphabet_size -> (batch_size, max_sequence_length, alphabet_size)
        output_vectors = self.character_decoder(output)

        return output_vectors, labels

    def _calculate_loss(self, scores, labels):
        # score vector has to have a certain format for (2d-)loss fct (batch_size, alphabet_size, 1, max_seq_length)
        scores_in_correct_format = scores.permute(0, 2, 1).unsqueeze(2)
    
        # create target vector (batch_size, max_label_seq_length + 1)
        target = self.labels_to_char_indices(labels, for_input=False)

        target.unsqueeze_(1) # (batch_size, 1, max_label_seq_length + 1)

        return self.loss(scores_in_correct_format, target)

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.tensor:
        scores, labels = self.forward_pass(sentences)
        if scores == None:
            return torch.tensor([0.], requires_grad=True)
        return self._calculate_loss(scores, labels)

    def predict(self, sentences: Union[List[Sentence], Sentence],
                label_name='predicted',
                mini_batch_size: int = 16,
                embedding_storage_mode="None",
                return_loss=False,
                print_prediction=False):
        
        if isinstance(sentences, Sentence):
                sentences = [sentences]

        # filter empty sentences
        sentences = [sentence for sentence in sentences if len(sentence) > 0]
        if len(sentences) == 0:
            return sentences

        dataloader = DataLoader(dataset=SentenceDataset(sentences), batch_size=mini_batch_size)

        overall_loss = 0
        label_count = 0
        
        with torch.no_grad():
            
            # embedd sentences
            self.token_embedding.embed(sentences)
            
            for batch in dataloader:

                # stop if all sentences are empty
                if not batch:
                    continue

                # remove previously predicted labels of this type
                for sentence in batch:
                    sentence.remove_labels(label_name)

                #create list of tokens in batch
                tokens_in_batch = [token for sentence in batch for token in sentence]

                # create inital hidden state tensor for batch (num_layers, batch_size, hidden_size)
                hidden_states = self.emb_to_hidden(torch.stack(self.num_layers * [torch.stack([token.get_embedding() for token in tokens_in_batch])]))

                # input (batch_size, 1, input_size), first letter is special character <S>
                input_indices = torch.ones(len(tokens_in_batch), dtype=torch.int).to(flair.device)
                
                input_tensor = self.character_embedding(input_indices).unsqueeze(1)
                
                indices_list = []

                if return_loss:

                    labels = [token.get_tag(label_type=self.label_name).value if token.get_tag(
                        label_type=self.label_name).value else token.text for token in tokens_in_batch]

                    target = self.labels_to_char_indices(labels, for_input=False)

                    if target == None: # unknown characeter in sentence/batch
                        continue
                    target_length = target.size()[1]

                    label_count += len(tokens_in_batch)

                # vector that checks if for all words a <E> has been predicted
                check = torch.tensor(len(tokens_in_batch) * [False]).to(flair.device)

                # since we wait until the RNN outputs <E> it could happen that there is an endless loop if never a <E> is predicted
                # thus after max_length letters we stop the prediction
                max_length = 20 # TODO: define max_length dependend on batch

                for j in range(max_length):
                    output, hidden_states = self.rnn(input_tensor, hidden_states)
                    output_vectors = self.character_decoder(output)

                    # compute loss in each step and sum it up
                    if return_loss:
                        if j < target_length:
                            t = target[:,j]
                        else:  # its possible that the number of predicted letters exceeds the actual target word, in this case we use the dummy index 0 as target
                            t = torch.tensor(len(tokens_in_batch) * [0]).to(flair.device)

                        overall_loss += self.loss(output_vectors.squeeze(1), t)

                    # to get predictions, take letter with highest output probability
                    # TODO: Is there a better way to get the actual lettes than this greedy version? Prediction of all letters at the same time, dependent on each other?
                    out_probs = self.softmax(output_vectors)

                    probabilities, indices = out_probs.topk(1, 2)  # max prob along dimension 2
                    
                    indices = indices[:, 0, 0]
                    indices_list.append(indices)
                    
                    check += (indices == 2)
                    
                    if not (False in check):  # for every word in the batch at least one <E> has been predicted
                        break
                    
                    input_tensor = self.character_embedding(indices).unsqueeze(1)
                    
                idxs = torch.stack(indices_list, dim=1)

                for i in range(len(tokens_in_batch)):
                    word = ''
                    for k in range(max_length):
                        if idxs[i][k] == 2:
                            break
                        word+= self.char_dictionary.get_item_for_index(idxs[i][k])
                    if print_prediction:
                        print(word)
                    tokens_in_batch[i].add_tag(tag_type=label_name, tag_value=word)
                
            store_embeddings(sentences, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count
        
    def evaluate(
            self,
            sentences: Union[List[Sentence], Dataset],
            gold_label_type: str,
            out_path: Union[str, Path] = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 16,
            num_workers: int = 8,
            main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
            exclude_labels: List[str] = [],
            gold_label_dictionary: Optional[Dictionary] = None,
    ) -> Result:
        
        import numpy as np
        import sklearn

        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)

        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        with torch.no_grad():

            # loss calculation
            eval_loss = 0
            average_over = 0

            all_labels_and_predictions = []

            all_labels_dict = Dictionary(add_unk=True)
            all_label_names = []

            for batch in data_loader:

                # remove any previously predicted labels
                for sentence in batch:
                    for token in sentence:
                        token.remove_labels("predicted")

                # predict for batch
                loss_and_count = self.predict(batch,
                                              embedding_storage_mode=embedding_storage_mode,
                                              mini_batch_size=mini_batch_size,
                                              label_name='predicted',
                                              return_loss=True)

                average_over += loss_and_count[1]
                eval_loss += loss_and_count[0]

                # get the gold labels
                for sentence in batch:
                    if not sentence[0].get_labels('predicted'):  # sentence with unknown character
                        continue

                    sentence_labels_and_predictions = []
                    for token in sentence:
                        lemma = token.get_labels('lemma')[0].value
                        all_labels_dict.add_item(lemma)
                        all_label_names.append(lemma)
                        sentence_labels_and_predictions.append((lemma, token.get_labels('predicted')[0].value))

                    all_labels_and_predictions.append((sentence.to_plain_string(), sentence_labels_and_predictions))

                store_embeddings(batch, embedding_storage_mode)

            if not all_labels_and_predictions:
                raise RuntimeError('Nothing predicted in evaluate function. Do all given sentences contain unknown characters??')

            # write all_predicted_values to out_file if given
            if out_path:
                with open(Path(out_path), "w", encoding="utf-8") as outfile:
                    for tuple in all_labels_and_predictions:
                        outfile.write(tuple[0] + '\n') # the sentence
                        labels = [x[0] for x in tuple[1]]
                        predictions = [x[1] for x in tuple[1]]
                        outfile.write((' ').join(labels) + '\n')
                        outfile.write((' ').join(predictions) + '\n\n')

            y_true = []
            y_pred = []

            for tuple in all_labels_and_predictions:
                for lemma, prediction in tuple[1]:
                    y_true.append(all_labels_dict.get_idx_for_item(lemma))
                    y_pred.append(all_labels_dict.get_idx_for_item(prediction))

            # sort by number of occurences
            counter = Counter()
            counter.update(all_label_names)

            target_names = []
            corresponding_indices = []

            for label_name, count in counter.most_common():

                if label_name in exclude_labels: continue
                target_names.append(label_name)
                corresponding_indices.append(all_labels_dict.get_idx_for_item(label_name))

            classification = classification_report(
                y_true, y_pred, digits=4, target_names=target_names, zero_division=0, labels=corresponding_indices,
            )

            classification_report_dict = classification_report(
                y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True, labels=corresponding_indices,
            )

            accuracy = round(accuracy_score(y_true, y_pred), 4)

            precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
            recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
            micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

            detailed_result = (
                    "\nResults:"
                    f"\n- F-score (micro) {micro_f_score}"
                    f"\n- F-score (macro) {macro_f_score}"
                    f"\n- Accuracy {accuracy}"
                    "\n\nBy class:\n" + classification
            )

            # line for log file
            log_header = "PRECISION\tRECALL\tF1\tACCURACY"
            log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy}"

            if average_over > 0:
                eval_loss /= average_over

            result = Result(
                main_score=main_score,
                log_line=log_line,
                log_header=log_header,
                detailed_results=detailed_result,
                classification_report=classification_report_dict,
                loss=eval_loss
            )

            return result

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.token_embedding,
            "input_size": self.input_size,
            "hidden_size":  self.hidden_size,
            "path_to_char_dict": self.path_to_char_dict,
            "num_layers_in_rnn": self.num_layers,
            "label_name": self.label_name
        }

        return model_state

    def _init_model_with_state_dict(state):
        model = Lemmatizer(
            token_embedding=state["embeddings"],
            input_size=state["input_size"],
            hidden_size=state["hidden_size"],
            path_to_char_dict=state["path_to_char_dict"],
            num_layers_in_rnn=state["num_layers_in_rnn"],
            label_name=state["label_name"]
        )
        model.load_state_dict(state["state_dict"])
        return model
