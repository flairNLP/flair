from flair.data import Dictionary
from flair.training_utils import convert_labels_to_one_hot


def test_convert_labels_to_one_hot():
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("class-1")
    label_dict.add_item("class-2")
    label_dict.add_item("class-3")

    one_hot = convert_labels_to_one_hot([["class-2"]], label_dict)

    assert one_hot[0][0] == 0
    assert one_hot[0][1] == 1
    assert one_hot[0][2] == 0
