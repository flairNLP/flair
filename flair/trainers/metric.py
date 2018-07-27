class Metric(object):

    def __init__(self, name):
        self.name = name

        self._tp = 0.0
        self._fp = 0.0
        self._tn = 0.0
        self._fn = 0.0

    def tp(self):
        self._tp += 1

    def tn(self):
        self._tn += 1

    def fp(self):
        self._fp += 1

    def fn(self):
        self._fn += 1

    def precision(self):
        if self._tp + self._fp > 0:
            return self._tp / (self._tp + self._fp)
        return 0.0

    def recall(self):
        if self._tp + self._fn > 0:
            return self._tp / (self._tp + self._fn)
        return 0.0

    def f_score(self):
        if self.precision() + self.recall() > 0:
            return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
        return 0.0

    def accuracy(self):
        if self._tp + self._tn + self._fp + self._fn > 0:
            return (self._tp + self._tn) / (self._tp + self._tn + self._fp + self._fn)
        return 0.0

    def __str__(self):
        return '{0:<20}\tprecision: {1:.4f} - recall: {2:.4f} - accuracy: {3:.4f} - f1-score: {4:.4f}'.format(
            self.name, self.precision(), self.recall(), self.accuracy(), self.f_score())

    def print(self):
        print('{0:<20}\tprecision: {1:.4f} - recall: {2:.4f} - accuracy: {3:.4f} - f1-score: {4:.4f}'.format(
                self.name, self.precision(), self.recall(), self.accuracy(), self.f_score()))
