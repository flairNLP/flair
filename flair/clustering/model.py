from abc import abstractmethod


class Clustering:
    @abstractmethod
    def fit(self, vectors: list) -> list:
        pass

    def get_label_list(self, sentences) -> list:
        return list(map(lambda e: int(e.get_labels("cluster")[0].value), sentences))
