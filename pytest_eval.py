import flair
from pathlib import Path
import os

from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings, FlairEmbeddings
from flair.trainers import ModelTrainer
from tests.conftest import results_base_path

def resources_path():
    return "tests/resources"

def tasks_base_path():
    return os.path.join(resources_path(), "tasks")

def results_base_path():
    return os.path.join(resources_path(),"results")

def moin():
    corpus = flair.datasets.ColumnCorpus(
        data_folder=os.path.join(tasks_base_path(), "fashion"), column_format={0: "text", 3: "ner"}
    )

    turian_embeddings = WordEmbeddings("turian")
    flair_embeddings = FlairEmbeddings("news-forward-fast")

    tag_dictionary = corpus.make_tag_dictionary("ner")

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=turian_embeddings,
        tag_dictionary=tag_dictionary,
        tag_type="ner",
        use_crf=False,
    )

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        results_base_path(),
        learning_rate=0.1,
        mini_batch_size=2,
        max_epochs=2,
        shuffle=False,
    )

    del trainer, tagger, tag_dictionary, corpus

if __name__ == "__main__":
    moin()