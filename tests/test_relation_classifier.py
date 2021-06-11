import shutil

from flair.data import Sentence
from flair.embeddings import (
    TransformerWordEmbeddings
)
from flair.models import RelationClassifier
from flair.trainers import ModelTrainer
from flair.datasets.relation_extraction import CoNLLUCorpus


# @pytest.mark.integration
def test_train_load_use_classifier(results_base_path, tasks_base_path):
    corpus = CoNLLUCorpus(
        data_folder=tasks_base_path / "conllu",
        train_file="train.conllu",
        dev_file="train.conllu",
        test_file="train.conllu",
    )

    relation_label_dict = corpus.make_relation_label_dictionary(label_type="label")

    embeddings = TransformerWordEmbeddings()

    model: RelationClassifier = RelationClassifier(
        hidden_size=64,
        token_embeddings=embeddings,
        label_dictionary=relation_label_dict,
        label_type="label",
        span_label_type="ner",
    )

    # initialize trainer
    trainer: ModelTrainer = ModelTrainer(model, corpus)

    trainer.train(
        results_base_path,
        learning_rate=0.1,
        mini_batch_size=2,
        max_epochs=3,
        shuffle=False,
    )

    del trainer, model, relation_label_dict, corpus

    loaded_model: RelationClassifier = RelationClassifier.load(
        results_base_path / "final-model.pt"
    )

    sentence = Sentence(["Apple", "was", "founded", "by", "Steve", "Jobs", "."])
    for token, tag in zip(sentence.tokens, ["B-ORG", "O", "O", "O", "B-PER", "I-PER", "O"]):
        token.set_label("ner", tag)

    # sentence = Sentence("I love Berlin")
    # sentence_empty = Sentence("       ")

    loaded_model.predict(sentence)

    print("relations: ", sentence.relations)

    assert 1 == 0

    # loaded_model.predict([sentence, sentence_empty])
    # loaded_model.predict([sentence_empty])

    # clean up results directory
    shutil.rmtree(results_base_path)
    del loaded_model
