from flair.data import Dictionary
from flair.training_utils import convert_labels_to_one_hot, Metric


def test_metric_without_classes():
    metric = Metric('Test')

    for i in range(0, 5):
        metric.add_tp()
    for i in range(0, 20):
        metric.add_tn()
    for i in range(0, 10):
        metric.add_fp()
    for i in range(0, 5):
        metric.add_fn()

    assert(metric.precision() == 0.3333)
    assert(metric.recall() == 0.5)
    assert(metric.f_score() == 0.4)
    assert(metric.accuracy() == 0.625)


def test_metric_with_classes():
    metric = Metric('Test')

    for i in range(0, 5):
        metric.add_tp('class-1')
    for i in range(0, 5):
        metric.add_tp('class-2')
    for i in range(0, 20):
        metric.add_tn('class-1')
    for i in range(0, 5):
        metric.add_tn('class-2')
    for i in range(0, 10):
        metric.add_fp('class-1')
    for i in range(0, 5):
        metric.add_fp('class-2')
    for i in range(0, 5):
        metric.add_fn('class-1')
    for i in range(0, 5):
        metric.add_fn('class-2')

    assert(metric.precision('class-1') == 0.3333)
    assert(metric.precision('class-2') == 0.5)
    assert(metric.precision() == 0.0)
    assert(metric.recall('class-1') == 0.5)
    assert(metric.recall('class-2') == 0.5)
    assert(metric.recall() == 0.0)
    assert(metric.f_score('class-1') == 0.4)
    assert(metric.f_score('class-2') == 0.5)
    assert(metric.f_score() == 0.0)
    assert(metric.accuracy('class-1') == 0.625)
    assert(metric.accuracy('class-2') == 0.5)
    assert(metric.accuracy() == 0.0)

    assert(metric.micro_avg_f_score() == 0.4444)
    assert(metric.macro_avg_f_score() == 0.4166)
    assert(metric.micro_avg_accuracy() == 0.5833)
    assert(metric.macro_avg_accuracy() == 0.5625)


def test_convert_labels_to_one_hot():
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item('class-1')
    label_dict.add_item('class-2')
    label_dict.add_item('class-3')

    one_hot = convert_labels_to_one_hot([['class-2']], label_dict)

    assert(one_hot[0][0] == 0)
    assert(one_hot[0][1] == 1)
    assert(one_hot[0][2] == 0)
