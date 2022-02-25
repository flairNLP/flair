from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import RelationExtractor
from flair.trainers import ModelTrainer


# @pytest.mark.integration
def test_train_load_use_classifier(results_base_path, tasks_base_path):
    corpus = ColumnCorpus(
        data_folder=tasks_base_path / "conllu",
        train_file="train.conllup",
        dev_file="train.conllup",
        test_file="train.conllup",
        column_format={1: "text", 2: "pos", 3: "ner"},
    )

    relation_label_dict = corpus.make_label_dictionary(label_type="relation")

    embeddings = TransformerWordEmbeddings()

    model: RelationExtractor = RelationExtractor(
        embeddings=embeddings,
        label_dictionary=relation_label_dict,
        label_type="relation",
        entity_label_type="ner",
        train_on_gold_pairs_only=True,
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

    loaded_model: RelationExtractor = RelationExtractor.load(results_base_path / "final-model.pt")
    loaded_model.train_on_gold_pairs_only = False

    sentence = Sentence(["Apple", "was", "founded", "by", "Steve", "Jobs", "."])
    sentence[0:1].add_label("ner", "ORG")
    sentence[4:6].add_label("ner", "PER")

    loaded_model.predict(sentence)

    assert "founded_by" == sentence.get_labels("relation")[0].value

    del loaded_model
