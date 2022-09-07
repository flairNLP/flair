from torch import nn
import torch
from flair.data import Sentence
from flair.data import Dictionary


class RNNRegressor(nn.Module):
    def __init__(self,
                 char_embedding_dim: int = 25,
                 hidden_size_char: int = 50,
                 output_size: int = 4,
                 device = torch.device("cpu")):
        super(RNNRegressor, self).__init__()
        self.char_dictionary = Dictionary.load("common-chars")
        self.device = device

        self.char_embedding_dim: int = char_embedding_dim
        self.hidden_size_char: int = hidden_size_char
        self.char_embedding = torch.nn.Embedding(len(self.char_dictionary.item2idx), self.char_embedding_dim)

        self.char_rnn = torch.nn.LSTM(
            self.char_embedding_dim,
            self.hidden_size_char,
            num_layers=1,
            bidirectional=True,
        )

        self.output_size = output_size
        self.linear = nn.Linear(2 * self.hidden_size_char, self.output_size)

        self.to(self.device)

    def forward(self, batch):
        strings = list(batch[0]) # when using DataSet that already returns Flair Sentences
        target_vectors = batch[1]

        # convert strings to sequence of char ids with mapping dict:
        strings_as_char_ids = []
        for s in strings:
            char_indices = [self.char_dictionary.get_idx_for_item(char) for char in s]
            strings_as_char_ids.append(char_indices)

        sorted_by_length = sorted(strings_as_char_ids, key=lambda p: len(p), reverse=True)
        # keep mapping to original ordering
        d = {}
        for i, ci in enumerate(strings_as_char_ids):
            for j, cj in enumerate(sorted_by_length):
                if ci == cj:
                    d[j] = i
                    continue

        chars2_length = [len(c) for c in sorted_by_length]
        longest = max(chars2_length)
        mask = torch.zeros(
            (len(sorted_by_length), longest),
            dtype=torch.long,
            device=self.device,
        )

        for i, c in enumerate(sorted_by_length):
            mask[i, : chars2_length[i]] = torch.tensor(c, dtype=torch.long, device=self.device)

        # chars for rnn processing
        chars = mask
        embedded = self.char_embedding(chars).transpose(0, 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, chars2_length)  # type: ignore

        output, (hn, cn) = self.char_rnn(packed)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output)
        output = output.transpose(0, 1)  # so now (BS, seqlen, hidden_dim)

        # now: what to use as input to linear output layer?
        #input_linear_sorted = torch.cat((hn[0], hn[1]), dim=1) # concat last in both directions

        #print(output[:,-1,:] == torch.cat((hn[0], hn[1]), dim=1)) # shouldn't this be the same? hidden state of last character in sequence (2*d because bidirectionality)
        #input_linear_sorted = output[:, -1, :] # hidden state of last (or better first?) char in sequence, made out of the two directions (bidirectional lstm)

        # or better: take embedding of real last character?
        last_char_embedding = []
        for i, index in enumerate(output_lengths):
            last_char_embedding.append(output[i, index - 1]) # take the embedding of the last real character in sequence (so before the zero padding)
        input_linear_sorted = torch.stack(last_char_embedding, dim=0)

        # important: get back to original ordering with the help of d!
        input_linear_original = input_linear_sorted.clone()
        for i in range(input_linear_original.size(0)):
            input_linear_original[d[i]] = input_linear_sorted[i]

        output = self.linear(input_linear_original)

        return output, target_vectors

    def predict(self, strings):

        if not isinstance(strings, list):
            strings = [strings]

        with torch.no_grad():
            self.eval() # TODO: necessary?
            dummy_targets = torch.zeros(len(strings)) # dummy because forward needs target_vectors at the moment...
            batch = [strings, dummy_targets]
            output, _ = self.forward(batch)
        return output#.detach().cpu().numpy()


# this version was using Flair's CharacterEmbeddings. In principle the same thing but needed Flair Sentences which was time consuming
# class RNNRegressorOld(nn.Module):
#     def __init__(self, character_embeddings, output_size, device = torch.device("cpu")):
#         super(RNNRegressorOld, self).__init__()
#         self.device = device
#
#         self.character_embeddings = character_embeddings
#         #self.char_embed_size = self.character_embeddings.char_embedding_dim
#         self.hidden_size = self.character_embeddings.hidden_size_char
#
#         self.output_size = output_size
#         self.linear = nn.Linear(2*self.hidden_size, self.output_size)
#
#         self.to(self.device)
#
#     def forward(self, batch):
#         strings_as_sentences = [b[0] for b in batch] # when using DataSet that already returns Flair Sentences
#         #strings_as_sentences = [Sentence([s], use_tokenizer=False) for s in batch[0]] # when using DataSet that returns strings, not Sentences
#
#         target_vectors = torch.stack([b[1] for b in batch])
#
#         self.character_embeddings.embed(strings_as_sentences)
#
#         embedding_names = self.character_embeddings.get_names()
#
#         # this would need to be changed when every token gets a representation
#         #strings_as_sentences = [Sentence(s, use_tokenizer=False) for s in strings]
#
#         text_embedding_list = [s.tokens[0].get_embedding(embedding_names).unsqueeze(0) for s in strings_as_sentences]
#         text_embedding_tensor = torch.cat(text_embedding_list, 0)
#
#         output = self.linear(text_embedding_tensor)
#
#         return output, target_vectors#, weighting_info
#
#     def predict(self, strings):
#         with torch.no_grad():
#             #self.eval() # TODO: necessary?
#             dummy_targets = torch.zeros(len(strings)) # dummy because forward needs target_vectors at the moment...
#             dummy_weighting = torch.zeros(2)
#             batch = [strings, dummy_targets, dummy_weighting]
#             output, _, _ = self.forward(batch)
#         return output#.detach().cpu().numpy()










