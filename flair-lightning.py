from argparse import ArgumentParser
from typing import Optional


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.embeddings.base import Embeddings
from flair.data import Dictionary


class FlairData(pl.LightningDataModule):
    def __init__(
        self,
        corpus: Corpus,
        document_embeddings: Embeddings,
        label_dictionary: Dictionary,
        batch_size: int = 128,
    ):
        super().__init__()
        self.corpus = corpus
        self.document_embeddings = document_embeddings
        self.label_dictionary = label_dictionary
        self.batch_size = batch_size


    def transfer_batch_to_device(self, batch, device):
        self.document_embeddings.embed(batch)
        embedding_names = self.document_embeddings.get_names()

        text_embedding_list = [
            sentence.get_embedding(embedding_names).unsqueeze(0) for sentence in batch
        ]

        indices = [
            torch.LongTensor(
                [
                    self.label_dictionary.get_idx_for_item(label.value)
                    for label in sentence.get_labels()
                ]
            )
            for sentence in batch
        ]

        return (torch.cat(text_embedding_list, 0), torch.cat(indices, 0).to(device))

    def train_dataloader(self):
        return DataLoader(self.corpus.train.dataset,
                          collate_fn=list,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True,
                         )

    def val_dataloader(self):
        return DataLoader(self.corpus.dev.dataset,
                          collate_fn=list,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.corpus.test.dataset,
                          collate_fn=list,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=8)


class FlairModel(pl.LightningModule):
    def __init__(self, document_embeddings, label_dictionary, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.label_dictionary = label_dictionary
        self.document_embeddings = document_embeddings
        self.decoder = nn.Linear(
            self.document_embeddings.embedding_length, len(self.label_dictionary)
        )
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        scores = self(embeddings)

        return self.loss_function(scores, labels)

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        scores = self(embeddings)

        self.log('val_loss', self.loss_function(scores, labels), on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def cli_main():
    corpus = TREC_6()
    document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True, batch_size=128)
    label_dictionary = corpus.make_label_dictionary()

    trainer = pl.Trainer(gpus=1, accelerator="horovod")

    model = FlairModel(document_embeddings=document_embeddings,
                   label_dictionary=label_dictionary, learning_rate=1e-3)

    dm = FlairData(corpus=corpus,
               document_embeddings=model.document_embeddings,
               label_dictionary=model.label_dictionary)


    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()

