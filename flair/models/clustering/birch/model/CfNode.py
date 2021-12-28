from birch.model.ClusteringFeature import ClusteringFeature


class CfNode:
    def __init__(self):
        self.cfs = []
        self.isLeaf = False
        self.parent = None

    def sumAllCfs(self) -> ClusteringFeature:
        cf = ClusteringFeature()

        for help in self.cfs:
            cf.absorbCf(help)

        return cf
