import flair
import torch
import torch.nn as nn
from typing import List, Union
from flair.training_utils import clear_embeddings
from flair.data import Sentence, Label
import logging

log = logging.getLogger('flair')

class TextRegressor(flair.models.TextClassifier):
  
    def __init__(self,
                 document_embeddings: flair.embeddings.DocumentEmbeddings,
                 label_dictionary: flair.data.Dictionary,
                 multi_label: bool):

        super(TextRegressor, self).__init__(document_embeddings=document_embeddings, label_dictionary=flair.data.Dictionary(), multi_label=multi_label)

        log.info('Using REGRESSION - experimental')

        self.loss_function = nn.MSELoss()

    def _labels_to_indices(self, sentences: List[Sentence]):
        indices = [
            torch.FloatTensor([float(label.value) for label in sentence.labels])
            for sentence in sentences
        ]

        vec = torch.cat(indices, 0)
        if torch.cuda.is_available():
            vec = vec.cuda()

        return vec
      
    def forward_labels_and_loss(self, sentences: Union[Sentence, List[Sentence]]) -> (List[List[float]], torch.tensor):
        scores = self.forward(sentences)
        loss = self._calculate_loss(scores, sentences)
        return scores, loss

    def predict(self, sentences: Union[Sentence, List[Sentence]], mini_batch_size: int = 32) -> List[Sentence]:

        with torch.no_grad():
            if type(sentences) is Sentence:
                sentences = [sentences]

            filtered_sentences = self._filter_empty_sentences(sentences)

            batches = [filtered_sentences[x:x + mini_batch_size] for x in range(0, len(filtered_sentences), mini_batch_size)]

            for batch in batches:
                scores = self.forward(batch)

                for (sentence, score) in zip(batch, scores.tolist()):
                  sentence.labels = [Label(value=str(score[0]))]

                clear_embeddings(batch)

            return sentences
