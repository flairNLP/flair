from operator import itemgetter
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pytest
from torch.utils.data import Dataset

from flair.data import Relation, Sentence
from flair.datasets import ColumnCorpus, DataLoader
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import RelationClassifier
from flair.models.relation_classifier_model import EncodedSentence
from flair.trainers import ModelTrainer


@pytest.fixture(scope="function")
def corpus(tasks_base_path: Path) -> ColumnCorpus:
    return ColumnCorpus(
        data_folder=tasks_base_path / "conllu",
        train_file="train.conllup",
        dev_file="train.conllup",
        test_file="train.conllup",
        column_format={1: "text", 2: "pos", 3: "ner"},
    )


class TestTransform:
    @staticmethod
    def check_transformation_correctness(
        split: Optional[Dataset],
        ground_truth: Set[Tuple[str, Tuple[str, ...]]],
    ) -> None:
        """Ground truth is a set of tuples of (<Sentence Text>, <Relation Label Values>)"""
        assert split is not None

        data_loader = DataLoader(split, batch_size=1, num_workers=0)
        assert all(isinstance(sentence, EncodedSentence) for sentence in map(itemgetter(0), data_loader))
        assert {
            (sentence.to_tokenized_string(), tuple(label.value for label in sentence.get_labels("relation")))
            for sentence in map(itemgetter(0), data_loader)
        } == ground_truth

    def test_transform_corpus_with_cross_augmentation(self, corpus: ColumnCorpus) -> None:
        label_dictionary = corpus.make_label_dictionary("relation")
        embeddings = TransformerDocumentEmbeddings(model="distilbert-base-uncased", layers="-1", fine_tune=True)
        model: RelationClassifier = RelationClassifier(
            document_embeddings=embeddings,
            label_dictionary=label_dictionary,
            label_type="relation",
            entity_label_types="ner",
            entity_pair_labels={  # Define valid entity pair combinations, used as relation candidates
                ("ORG", "PER"),  # founded_by
                ("LOC", "PER"),  # place_of_birth
            },
            cross_augmentation=True,
        )

        transformed_corpus = model.transform_corpus(corpus)

        # Check sentence masking and relation label annotation on
        # training, validation and test dataset (in this test they are the same)
        ground_truth: Set[Tuple[str, Tuple[str, ...]]] = {
            # Entity pair permutations of: "Larry Page and Sergey Brin founded Google ."
            ("[[T-Larry Page]] and Sergey Brin founded [[H-Google]] .", ("founded_by",)),
            ("Larry Page and [[T-Sergey Brin]] founded [[H-Google]] .", ("founded_by",)),
            # Entity pair permutations of: "Microsoft was founded by Bill Gates ."
            ("[[H-Microsoft]] was founded by [[T-Bill Gates]] .", ("founded_by",)),
            # Entity pair permutations of: "Konrad Zuse was born in Berlin on 22 June 1910 ."
            ("[[T-Konrad Zuse]] was born in [[H-Berlin]] on 22 June 1910 .", ("place_of_birth",)),
            # Entity pair permutations of: "Joseph Weizenbaum , a professor at MIT , was born in Berlin , Germany."
            ("[[T-Joseph Weizenbaum]] , a professor at [[H-MIT]] , was born in Berlin , Germany .", ("O",)),
            (
                "[[T-Joseph Weizenbaum]] , a professor at MIT , was born in [[H-Berlin]] , Germany .",
                ("place_of_birth",),
            ),
            (
                "[[T-Joseph Weizenbaum]] , a professor at MIT , was born in Berlin , [[H-Germany]] .",
                ("place_of_birth",),
            ),
        }
        for split in (transformed_corpus.train, transformed_corpus.dev, transformed_corpus.test):
            TestTransform.check_transformation_correctness(split, ground_truth)

    def test_transform_corpus_without_cross_augmentation(self, corpus: ColumnCorpus) -> None:
        label_dictionary = corpus.make_label_dictionary("relation")
        embeddings = TransformerDocumentEmbeddings(model="distilbert-base-uncased", layers="-1", fine_tune=True)
        model: RelationClassifier = RelationClassifier(
            document_embeddings=embeddings,
            label_dictionary=label_dictionary,
            label_type="relation",
            entity_label_types="ner",
            entity_pair_labels={  # Define valid entity pair combinations, used as relation candidates
                ("ORG", "PER"),  # founded_by
                ("LOC", "PER"),  # place_of_birth
            },
            cross_augmentation=False,
        )

        transformed_corpus = model.transform_corpus(corpus)

        # Check sentence masking and relation label annotation on
        # training, validation and test dataset (in this test they are the same)
        ground_truth: Set[Tuple[str, Tuple[str, ...]]] = {
            # Entity pair permutations of: "Larry Page and Sergey Brin founded Google ."
            ("[[T-Larry Page]] and Sergey Brin founded [[H-Google]] .", ("founded_by",)),
            ("Larry Page and [[T-Sergey Brin]] founded [[H-Google]] .", ("founded_by",)),
            # Entity pair permutations of: "Microsoft was founded by Bill Gates ."
            ("[[H-Microsoft]] was founded by [[T-Bill Gates]] .", ("founded_by",)),
            # Entity pair permutations of: "Konrad Zuse was born in Berlin on 22 June 1910 ."
            ("[[T-Konrad Zuse]] was born in [[H-Berlin]] on 22 June 1910 .", ("place_of_birth",)),
            # Entity pair permutations of: "Joseph Weizenbaum , a professor at MIT , was born in Berlin , Germany ."
            (
                "[[T-Joseph Weizenbaum]] , a professor at MIT , was born in [[H-Berlin]] , Germany .",
                ("place_of_birth",),
            ),
            (
                "[[T-Joseph Weizenbaum]] , a professor at MIT , was born in Berlin , [[H-Germany]] .",
                ("place_of_birth",),
            ),
        }
        for split in (transformed_corpus.train, transformed_corpus.dev, transformed_corpus.test):
            TestTransform.check_transformation_correctness(split, ground_truth)


@pytest.mark.integration
def test_train_load_use_relation_classifier(results_base_path: Path, tasks_base_path: Path) -> None:
    # ---- Test Training ----
    # Hyperparameters
    transformer: str = "distilbert-base-uncased"
    num_epochs: int = 25
    learning_rate: float = 4e-5
    mini_batch_size: int = 4

    # Step 1: Create the training data
    # The relation extractor is *not* trained end-to-end.
    # A corpus for training the relation extractor requires annotated entities and relations.
    corpus: ColumnCorpus = ColumnCorpus(
        data_folder=tasks_base_path / "conllu",
        train_file="train.conllup",
        dev_file="train.conllup",
        test_file="train.conllup",
        column_format={1: "text", 2: "pos", 3: "ner"},
    )

    # Step 2: Make the label dictionary from the corpus
    label_dictionary = corpus.make_label_dictionary("relation")

    # Step 3: Initialize fine-tunable transformer embeddings
    embeddings = TransformerDocumentEmbeddings(model=transformer, layers="-1", fine_tune=True)

    # Step 4: Initialize relation classifier
    model: RelationClassifier = RelationClassifier(
        document_embeddings=embeddings,
        label_dictionary=label_dictionary,
        label_type="relation",
        entity_label_types="ner",
        entity_pair_labels={  # Define valid entity pair combinations, used as relation candidates
            ("ORG", "PER"),  # founded_by
            ("LOC", "PER"),  # place_of_birth
        },
        allow_unk_tag=False,
    )

    # Step 5: Initialize trainer on transformed corpus
    trainer: ModelTrainer = ModelTrainer(model=model, corpus=model.transform_corpus(corpus))

    # Step 6: Run fine-tuning
    trainer.fine_tune(
        results_base_path,
        max_epochs=num_epochs,
        learning_rate=learning_rate,
        mini_batch_size=mini_batch_size,
        main_evaluation_metric=("macro avg", "f1-score"),
        shuffle=False,
    )

    # Clean-up
    del trainer, model, label_dictionary, corpus

    # ---- Test Loading and Predicting ----
    # Step 1: Load trained relation extraction model
    loaded_model: RelationClassifier = RelationClassifier.load(results_base_path / "final-model.pt")

    # Step 2: Create sentences with entity annotations (as these are required by the relation extraction model)
    # In production, use another sequence tagger model to tag the relevant entities.
    sentence: Sentence = Sentence(
        [
            "Intel",
            "was",
            "founded",
            "on",
            "July",
            "18",
            ",",
            "1968",
            ",",
            "by",
            "semiconductor",
            "pioneers",
            "Gordon",
            "Moore",
            "and",
            "Robert",
            "Noyce",
            ".",
        ]
    )
    sentence[:1].add_label(typename="ner", value="ORG", score=1.0)  # Intel -> ORG
    sentence[12:14].add_label(typename="ner", value="PER", score=1.0)  # Gordon Moore -> PER
    sentence[15:17].add_label(typename="ner", value="PER", score=1.0)  # Robert Noyce -> PER

    # Step 3: Predict
    loaded_model.predict(sentence)

    relations: List[Relation] = sentence.get_relations("relation")
    assert len(relations) == 2

    # Intel ----founded_by---> Gordon Moore
    assert [label.value for label in relations[0].labels] == ["founded_by"]
    assert (
        relations[0].unlabeled_identifier == Relation(first=sentence[:1], second=sentence[12:14]).unlabeled_identifier
    )

    # Intel ----founded_by---> Robert Noyce
    assert [label.value for label in relations[1].labels] == ["founded_by"]
    assert (
        relations[1].unlabeled_identifier == Relation(first=sentence[:1], second=sentence[15:17]).unlabeled_identifier
    )

    # Clean-up
    del loaded_model
