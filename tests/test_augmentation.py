from flair.data import Sentence
from flair.models.sequence_tagger_model import EntityTypeTaskPromptAugmentationStrategy, AugmentedSentence


def test_entity_type_task_prompt_augmentation_single_type():
    strategy = EntityTypeTaskPromptAugmentationStrategy(["genes"])

    sent = Sentence("This is a test sentence.")
    aug_sent = strategy.augment_sentence(sent)

    assert isinstance(aug_sent, AugmentedSentence)
    assert aug_sent.text.startswith("[ Tag genes ] ")
    assert len(aug_sent) == 10


def test_entity_type_task_prompt_augmentation_two_types():
    strategy = EntityTypeTaskPromptAugmentationStrategy(["genes", "diseases"])

    sent = Sentence("This is a test sentence.")
    aug_sent = strategy.augment_sentence(sent)

    assert isinstance(aug_sent, AugmentedSentence)
    assert aug_sent.text.startswith("[ Tag genes and diseases ] ")
    assert len(aug_sent) == 12


def test_entity_type_task_prompt_augmentation_multiple_types():
    strategy = EntityTypeTaskPromptAugmentationStrategy(["genes", "diseases", "chemicals"])

    sent = Sentence("This is a test sentence.")
    aug_sent = strategy.augment_sentence(sent)

    assert isinstance(aug_sent, AugmentedSentence)
    assert aug_sent.text.startswith("[ Tag genes, diseases and chemicals ] ")
    assert len(aug_sent) == 13


def test_entity_type_task_prompt_augmentation_label_transfer():
    strategy = EntityTypeTaskPromptAugmentationStrategy(["genes"])

    sent = Sentence("This is a test sentence.")
    sent[0:2].add_label("ner", "test", 1.0)
    sent[3:4].add_label("foo", "test", 1.0)

    aug_sent = strategy.augment_sentence(sent, "ner")

    assert isinstance(aug_sent, AugmentedSentence)
    assert aug_sent.text.startswith("[ Tag genes ] ")
    assert len(aug_sent.get_labels("foo")) == 0

    ner_labels = aug_sent.get_labels("ner")
    assert len(ner_labels) == 1
    assert len(ner_labels[0].data_point.tokens) == 2
    assert ner_labels[0].data_point.text == "This is"
    assert ner_labels[0].data_point.tokens[0].idx == 5
    assert ner_labels[0].data_point.tokens[-1].idx == 6
