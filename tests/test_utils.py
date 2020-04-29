from flair.data import Dictionary
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import convert_labels_to_one_hot, Metric


def test_metric_get_classes():
    metric = Metric("Test")

    metric.add_fn("class-1")
    metric.add_fn("class-3")
    metric.add_tn("class-1")
    metric.add_tp("class-2")

    assert 3 == len(metric.get_classes())
    assert "class-1" in metric.get_classes()
    assert "class-2" in metric.get_classes()
    assert "class-3" in metric.get_classes()


# def test_multiclass_metrics():
#
#     metric = Metric("Test")
#     available_labels = ["A", "B", "C"]
#
#     predictions = ["A", "B"]
#     true_values = ["A"]
#     TextClassifier._evaluate_sentence_for_text_classification(
#         metric, available_labels, predictions, true_values
#     )
#
#     predictions = ["C", "B"]
#     true_values = ["A", "B"]
#     TextClassifier._evaluate_sentence_for_text_classification(
#         metric, available_labels, predictions, true_values
#     )
#
#     print(metric)


def test_metric_with_classes():
    metric = Metric("Test")

    metric.add_tp("class-1")
    metric.add_tn("class-1")
    metric.add_tn("class-1")
    metric.add_fp("class-1")

    metric.add_tp("class-2")
    metric.add_tn("class-2")
    metric.add_tn("class-2")
    metric.add_fp("class-2")

    for i in range(0, 10):
        metric.add_tp("class-3")
    for i in range(0, 90):
        metric.add_fp("class-3")

    metric.add_tp("class-4")
    metric.add_tn("class-4")
    metric.add_tn("class-4")
    metric.add_fp("class-4")

    print(metric)

    assert metric.precision("class-1") == 0.5
    assert metric.precision("class-2") == 0.5
    assert metric.precision("class-3") == 0.1
    assert metric.precision("class-4") == 0.5

    assert metric.recall("class-1") == 1
    assert metric.recall("class-2") == 1
    assert metric.recall("class-3") == 1
    assert metric.recall("class-4") == 1

    assert metric.accuracy() == metric.micro_avg_accuracy()
    assert metric.f_score() == metric.micro_avg_f_score()

    assert metric.f_score("class-1") == 0.6666666666666666
    assert metric.f_score("class-2") == 0.6666666666666666
    assert metric.f_score("class-3") == 0.18181818181818182
    assert metric.f_score("class-4") == 0.6666666666666666

    assert metric.accuracy("class-1") == 0.75
    assert metric.accuracy("class-2") == 0.75
    assert metric.accuracy("class-3") == 0.1
    assert metric.accuracy("class-4") == 0.75

    assert metric.micro_avg_f_score() == 0.21848739495798317
    assert metric.macro_avg_f_score() == 0.5454545454545454

    assert metric.micro_avg_accuracy() == 0.16964285714285715
    assert metric.macro_avg_accuracy() == 0.5875

    assert metric.precision() == 0.12264150943396226
    assert metric.recall() == 1


def test_convert_labels_to_one_hot():
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("class-1")
    label_dict.add_item("class-2")
    label_dict.add_item("class-3")

    one_hot = convert_labels_to_one_hot([["class-2"]], label_dict)

    assert one_hot[0][0] == 0
    assert one_hot[0][1] == 1
    assert one_hot[0][2] == 0
