from operator import itemgetter
from typing import Optional

import pytest
from torch.utils.data import Dataset

from flair.data import Relation, Sentence
from flair.datasets import ColumnCorpus, DataLoader
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import RelationClassifier
from flair.models.relation_classifier_model import (
    EncodedSentence,
    EncodingStrategy,
    EntityMarker,
    EntityMarkerPunct,
    EntityMask,
    TypedEntityMarker,
    TypedEntityMarkerPunct,
    TypedEntityMask,
)
from tests.model_test_utils import BaseModelTest

encoding_strategies: dict[EncodingStrategy, list[tuple[str, str]]] = {
    EntityMask(): [("[HEAD]", "[TAIL]") for _ in range(7)],
    TypedEntityMask(): [
        ("[HEAD-ORG]", "[TAIL-PER]"),
        ("[HEAD-ORG]", "[TAIL-PER]"),
        ("[HEAD-ORG]", "[TAIL-PER]"),
        ("[HEAD-LOC]", "[TAIL-PER]"),
        ("[HEAD-LOC]", "[TAIL-PER]"),
        ("[HEAD-LOC]", "[TAIL-PER]"),
        ("[HEAD-ORG]", "[TAIL-PER]"),
    ],
    EntityMarker(): [
        ("[HEAD] Google [/HEAD]", "[TAIL] Larry Page [/TAIL]"),
        ("[HEAD] Google [/HEAD]", "[TAIL] Sergey Brin [/TAIL]"),
        ("[HEAD] Microsoft [/HEAD]", "[TAIL] Bill Gates [/TAIL]"),
        ("[HEAD] Berlin [/HEAD]", "[TAIL] Konrad Zuse [/TAIL]"),
        ("[HEAD] Berlin [/HEAD]", "[TAIL] Joseph Weizenbaum [/TAIL]"),
        ("[HEAD] Germany [/HEAD]", "[TAIL] Joseph Weizenbaum [/TAIL]"),
        ("[HEAD] MIT [/HEAD]", "[TAIL] Joseph Weizenbaum [/TAIL]"),
    ],
    TypedEntityMarker(): [
        ("[HEAD-ORG] Google [/HEAD-ORG]", "[TAIL-PER] Larry Page [/TAIL-PER]"),
        ("[HEAD-ORG] Google [/HEAD-ORG]", "[TAIL-PER] Sergey Brin [/TAIL-PER]"),
        ("[HEAD-ORG] Microsoft [/HEAD-ORG]", "[TAIL-PER] Bill Gates [/TAIL-PER]"),
        ("[HEAD-LOC] Berlin [/HEAD-LOC]", "[TAIL-PER] Konrad Zuse [/TAIL-PER]"),
        ("[HEAD-LOC] Berlin [/HEAD-LOC]", "[TAIL-PER] Joseph Weizenbaum [/TAIL-PER]"),
        ("[HEAD-LOC] Germany [/HEAD-LOC]", "[TAIL-PER] Joseph Weizenbaum [/TAIL-PER]"),
        ("[HEAD-ORG] MIT [/HEAD-ORG]", "[TAIL-PER] Joseph Weizenbaum [/TAIL-PER]"),
    ],
    EntityMarkerPunct(): [
        ("@ Google @", "# Larry Page #"),
        ("@ Google @", "# Sergey Brin #"),
        ("@ Microsoft @", "# Bill Gates #"),
        ("@ Berlin @", "# Konrad Zuse #"),
        ("@ Berlin @", "# Joseph Weizenbaum #"),
        ("@ Germany @", "# Joseph Weizenbaum #"),
        ("@ MIT @", "# Joseph Weizenbaum #"),
    ],
    TypedEntityMarkerPunct(): [
        ("@ * ORG * Google @", "# ^ PER ^ Larry Page #"),
        ("@ * ORG * Google @", "# ^ PER ^ Sergey Brin #"),
        ("@ * ORG * Microsoft @", "# ^ PER ^ Bill Gates #"),
        ("@ * LOC * Berlin @", "# ^ PER ^ Konrad Zuse #"),
        ("@ * LOC * Berlin @", "# ^ PER ^ Joseph Weizenbaum #"),
        ("@ * LOC * Germany @", "# ^ PER ^ Joseph Weizenbaum #"),
        ("@ * ORG * MIT @", "# ^ PER ^ Joseph Weizenbaum #"),
    ],
}


class TestRelationClassifier(BaseModelTest):
    model_cls = RelationClassifier
    train_label_type = "relation"
    multiclass_prediction_labels = ["apple", "tv"]
    model_args = {
        "entity_label_types": "ner",
        "entity_pair_labels": {  # Define valid entity pair combinations, used as relation candidates
            ("ORG", "PER"),  # founded_by
            ("LOC", "PER"),  # place_of_birth
        },
        "allow_unk_tag": False,
    }
    training_args = {"max_epochs": 2, "learning_rate": 4e-4, "mini_batch_size": 4}
    finetune_instead_of_train = True

    @pytest.fixture()
    def corpus(self, tasks_base_path):
        return ColumnCorpus(
            data_folder=tasks_base_path / "conllu",
            train_file="train.conllup",
            dev_file="train.conllup",
            test_file="train.conllup",
            column_format={1: "text", 2: "pos", 3: "ner"},
        )

    @pytest.fixture()
    def embeddings(self):
        return TransformerDocumentEmbeddings(model="distilbert-base-uncased", layers="-1", fine_tune=True)

    def transform_corpus(self, model, corpus):
        return model.transform_corpus(corpus)

    @pytest.fixture()
    def example_sentence(self):
        sentence = Sentence(["Microsoft", "was", "found", "by", "Bill", "Gates"])
        sentence[:1].add_label(typename="ner", value="ORG", score=1.0)
        sentence[4:].add_label(typename="ner", value="PER", score=1.0)
        return sentence

    @pytest.fixture()
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
        relations: list[Relation] = predicted_training_example.get_relations("relation")
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
        ground_truth: set[tuple[str, tuple[str, ...]]],
    ) -> None:
        # Ground truth is a set of tuples of (<Sentence Text>, <Relation Label Values>)
        assert split is not None

        data_loader = DataLoader(split, batch_size=1)
        assert all(isinstance(sentence, EncodedSentence) for sentence in map(itemgetter(0), data_loader))
        assert {
            (sentence.to_tokenized_string(), tuple(label.value for label in sentence.get_labels("relation")))
            for sentence in map(itemgetter(0), data_loader)
        } == ground_truth

    @pytest.mark.parametrize(
        "cross_augmentation", [True, False], ids=["with_cross_augmentation", "without_cross_augmentation"]
    )
    @pytest.mark.parametrize(
        ("encoding_strategy", "encoded_entity_pairs"),
        encoding_strategies.items(),
        ids=[type(encoding_strategy).__name__ for encoding_strategy in encoding_strategies],
    )
    def test_transform_corpus(
        self,
        corpus: ColumnCorpus,
        embeddings: TransformerDocumentEmbeddings,
        cross_augmentation: bool,
        encoding_strategy: EncodingStrategy,
        encoded_entity_pairs: list[tuple[str, str]],
    ) -> None:
        label_dictionary = corpus.make_label_dictionary("relation")
        model: RelationClassifier = self.build_model(
            embeddings, label_dictionary, cross_augmentation=cross_augmentation, encoding_strategy=encoding_strategy
        )
        transformed_corpus = model.transform_corpus(corpus)

        # Check sentence masking and relation label annotation on
        # training, validation and test dataset (in this test the splits are the same)
        ground_truth: set[tuple[str, tuple[str, ...]]] = {
            # Entity pair permutations of: "Larry Page and Sergey Brin founded Google ."
            (f"{encoded_entity_pairs[0][1]} and Sergey Brin founded {encoded_entity_pairs[0][0]} .", ("founded_by",)),
            (f"Larry Page and {encoded_entity_pairs[1][1]} founded {encoded_entity_pairs[1][0]} .", ("founded_by",)),
            # Entity pair permutations of: "Microsoft was founded by Bill Gates ."
            (f"{encoded_entity_pairs[2][0]} was founded by {encoded_entity_pairs[2][1]} .", ("founded_by",)),
            # Entity pair permutations of: "Konrad Zuse was born in Berlin on 22 June 1910 ."
            (
                f"{encoded_entity_pairs[3][1]} was born in {encoded_entity_pairs[3][0]} on 22 June 1910 .",
                ("place_of_birth",),
            ),
            # Entity pair permutations of: "Joseph Weizenbaum , a professor at MIT , was born in Berlin , Germany."
            (
                f"{encoded_entity_pairs[4][1]} , a professor at MIT , "
                f"was born in {encoded_entity_pairs[4][0]} , Germany .",
                ("place_of_birth",),
            ),
            (
                f"{encoded_entity_pairs[5][1]} , a professor at MIT , "
                f"was born in Berlin , {encoded_entity_pairs[5][0]} .",
                ("place_of_birth",),
            ),
        }

        if cross_augmentation:
            # This sentence is only included if we transform the corpus with cross augmentation
            ground_truth.add(
                (
                    f"{encoded_entity_pairs[6][1]} , a professor at {encoded_entity_pairs[6][0]} , "
                    f"was born in Berlin , Germany .",
                    ("O",),
                )
            )

        for split in (transformed_corpus.train, transformed_corpus.dev, transformed_corpus.test):
            self.check_transformation_correctness(split, ground_truth)
