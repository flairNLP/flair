from abc import ABC, abstractmethod


class Clustering(ABC):
    @abstractmethod
    def cluster(self, vectors: list) -> list:
        pass

    def getLabelList(self, listSenctence) -> list:
        return list(map(lambda e: int(e.get_labels('cluster')[0].value), listSenctence))
