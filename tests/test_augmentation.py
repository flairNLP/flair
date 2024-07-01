from flair.data import Sentence
from flair.models.prefixed_tagger import EntityTypeTaskPromptAugmentationStrategy, PrefixedSentence


def test_entity_type_task_prompt_augmentation_single_type():
    strategy = EntityTypeTaskPromptAugmentationStrategy(["genes"])

    sent = Sentence("This is a test sentence.")
    aug_sent = strategy.augment_sentence(sent)

    assert isinstance(aug_sent, PrefixedSentence)
    assert aug_sent.text.startswith("[ Tag genes ] ")
    assert len(aug_sent) == 10


def test_entity_type_task_prompt_augmentation_two_types():
    strategy = EntityTypeTaskPromptAugmentationStrategy(["genes", "diseases"])

    sent = Sentence("This is a test sentence.")
    aug_sent = strategy.augment_sentence(sent)

    assert isinstance(aug_sent, PrefixedSentence)
    assert aug_sent.text.startswith("[ Tag genes and diseases ] ")
    assert len(aug_sent) == 12


def test_entity_type_task_prompt_augmentation_multiple_types():
    strategy = EntityTypeTaskPromptAugmentationStrategy(["genes", "diseases", "chemicals"])

    sent = Sentence("This is a test sentence.")
    aug_sent = strategy.augment_sentence(sent)

    assert isinstance(aug_sent, PrefixedSentence)
    assert aug_sent.text.startswith("[ Tag genes, diseases and chemicals ] ")
    assert len(aug_sent) == 13


def test_entity_type_task_prompt_augmentation_label_transfer():
    strategy = EntityTypeTaskPromptAugmentationStrategy(["genes"])

    sent = Sentence("This is a test sentence.")
    sent[0:2].add_label("ner", "test", 1.0)
    sent[3:4].add_label("foo", "test", 1.0)

    aug_sent = strategy.augment_sentence(sent, "ner")

    assert isinstance(aug_sent, PrefixedSentence)
    assert aug_sent.text.startswith("[ Tag genes ] ")
    assert len(aug_sent.get_labels("foo")) == 0

    ner_labels = aug_sent.get_labels("ner")
    assert len(ner_labels) == 1
    assert len(ner_labels[0].data_point.tokens) == 2
    assert ner_labels[0].data_point.text == "This is"
    assert ner_labels[0].data_point.tokens[0].idx == 5
    assert ner_labels[0].data_point.tokens[-1].idx == 6


def test_entity_type_task_prompt_augmentation_label_application():
    strategy = EntityTypeTaskPromptAugmentationStrategy(["genes"])

    sent = Sentence("TP53 - also known as tumour protein 53 - is an onco-gene.")

    aug_sent = strategy.augment_sentence(sent, "ner")
    aug_sent[4:5].add_label("predict", "gene", 1.0)
    aug_sent[9:12].add_label("predict", "gene", 1.0)
    aug_sent[5:6].add_label("not-predict", "gene", 1.0)

    strategy.apply_predictions(aug_sent, sent, "predict", "ner")

    ner_labels = sent.get_labels("ner")
    assert len(ner_labels) == 2

    assert ner_labels[0].data_point.text == "TP53"
    assert ner_labels[0].value == "gene"
    assert ner_labels[0].score == 1.0
    assert len(ner_labels[0].data_point.tokens) == 1
    assert ner_labels[0].data_point.tokens[0].idx == 1

    assert ner_labels[1].data_point.text == "tumour protein 53"
    assert ner_labels[1].value == "gene"
    assert ner_labels[1].score == 1.0
    assert len(ner_labels[1].data_point.tokens) == 3
    assert ner_labels[1].data_point.tokens[0].idx == 6
    assert ner_labels[1].data_point.tokens[-1].idx == 8


def test_entity_type_task_prompt_augmentation_label_application_label_in_tag():
    strategy = EntityTypeTaskPromptAugmentationStrategy(["genes"])

    sent = Sentence("TP53 - also known as tumour protein 53 - is an onco-gene.")

    aug_sent = strategy.augment_sentence(sent, "ner")
    aug_sent[2:4].add_label("predict", "gene", 1.0)  # Add label in tagging prompt

    strategy.apply_predictions(aug_sent, sent, "predict", "ner")

    ner_labels = sent.get_labels("ner")
    assert len(ner_labels) == 0
