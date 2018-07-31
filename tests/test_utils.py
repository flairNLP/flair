import pytest

from flair.data import Dictionary
from flair.training_utils import calculate_micro_avg_metric, calculate_class_metrics


@pytest.fixture
def init():
    y_true = [[0, 1, 1], [0, 0, 1], [1, 1, 0]]
    y_pred = [[0, 1, 1], [0, 0, 0], [1, 0, 0]]

    labels = Dictionary(add_unk=False)
    labels.add_item('class-1')
    labels.add_item('class-2')
    labels.add_item('class-3')

    return y_true, y_pred, labels


def test_calculate_micro_avg_metric():
    y_true, y_pred, labels = init()

    metric = calculate_micro_avg_metric(y_true, y_pred, labels)

    assert(3 == metric._tp)
    assert(0 == metric._fp)
    assert(4 == metric._tn)
    assert(2 == metric._fn)


def test_calculate_class_metrics():
    y_true, y_pred, labels = init()

    metrics = calculate_class_metrics(y_true, y_pred, labels)

    metrics_dict = {metric.name: metric for metric in metrics}

    assert(3 == len(metrics))

    assert(1 == metrics_dict['class-1']._tp)
    assert(0 == metrics_dict['class-1']._fp)
    assert(2 == metrics_dict['class-1']._tn)
    assert(0 == metrics_dict['class-1']._fn)

    assert(1 == metrics_dict['class-2']._tp)
    assert(0 == metrics_dict['class-2']._fp)
    assert(1 == metrics_dict['class-2']._tn)
    assert(1 == metrics_dict['class-2']._fn)

    assert(1 == metrics_dict['class-3']._tp)
    assert(0 == metrics_dict['class-3']._fp)
    assert(1 == metrics_dict['class-3']._tn)
    assert(1 == metrics_dict['class-3']._fn)

