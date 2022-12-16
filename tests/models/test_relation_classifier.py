from operator import itemgetter
from typing import List, Optional, Set, Tuple

import pytest
from torch.utils.data import Dataset

from flair.data import Relation, Sentence
from flair.datasets import ColumnCorpus, DataLoader
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import RelationClassifier
from flair.models.relation_classifier_model import EncodedSentence
from tests.model_test_utils import BaseModelTest


class TestRelationClassifier(BaseModelTest):
    model_cls = RelationClassifier
    train_label_type = "relation"
    multiclass_prediction_labels = ["apple", "tv"]
    model_args = dict(
        entity_label_types="ner",
        entity_pair_labels={  # Define valid entity pair combinations, used as relation candidates
            ("ORG", "PER"),  # founded_by
            ("LOC", "PER"),  # place_of_birth
        },
        allow_unk_tag=False,
    )
    training_args = dict(max_epochs=25, learning_rate=4e-5, mini_batch_size=4)
    finetune_instead_of_train = True

    @pytest.fixture
    def corpus(self, tasks_base_path):
        return ColumnCorpus(
            data_folder=tasks_base_path / "conllu",
            train_file="train.conllup",
            dev_file="train.conllup",
            test_file="train.conllup",
            column_format={1: "text", 2: "pos", 3: "ner"},
        )

    @pytest.fixture
    def embeddings(self):
        yield TransformerDocumentEmbeddings(model="distilbert-base-uncased", layers="-1", fine_tune=True)

    def transform_corpus(self, model, corpus):
        return model.transform_corpus(corpus)

    @pytest.fixture
    def example_sentence(self):
        sentence = Sentence(["Microsoft", "was", "found", "by", "Bill", "Gates"])
        sentence[:1].add_label(typename="ner", value="ORG", score=1.0)
        sentence[4:].add_label(typename="ner", value="PER", score=1.0)
        yield sentence

    @pytest.fixture
    def train_test_sentence(self):
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

        return sentence

    def assert_training_example(self, predicted_training_example):
        relations: List[Relation] = predicted_training_example.get_relations("relation")
        assert len(relations) == 2

        # Intel ----founded_by---> Gordon Moore
        assert [label.value for label in relations[0].labels] == ["founded_by"]
        assert (
            relations[0].unlabeled_identifier
            == Relation(
                first=predicted_training_example[:1], second=predicted_training_example[12:14]
            ).unlabeled_identifier
        )

        # Intel ----founded_by---> Robert Noyce
        assert [label.value for label in relations[1].labels] == ["founded_by"]
        assert (
            relations[1].unlabeled_identifier
            == Relation(
                first=predicted_training_example[:1], second=predicted_training_example[15:17]
            ).unlabeled_identifier
        )

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

    def test_transform_corpus_with_cross_augmentation(self, corpus: ColumnCorpus, embeddings) -> None:
        label_dictionary = corpus.make_label_dictionary("relation")
        model: RelationClassifier = self.build_model(embeddings, label_dictionary, cross_augmentation=True)
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
            self.check_transformation_correctness(split, ground_truth)

    def test_transform_corpus_without_cross_augmentation(self, corpus: ColumnCorpus, embeddings) -> None:
        label_dictionary = corpus.make_label_dictionary("relation")
        embeddings = TransformerDocumentEmbeddings(model="distilbert-base-uncased", layers="-1", fine_tune=True)
        model: RelationClassifier = self.build_model(embeddings, label_dictionary, cross_augmentation=False)

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
            self.check_transformation_correctness(split, ground_truth)
