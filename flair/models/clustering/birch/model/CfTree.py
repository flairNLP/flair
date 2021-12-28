import numpy as np
from birch.model import LeafNode
from birch.model.CfNode import CfNode
from birch.model.ClusteringFeature import ClusteringFeature
from birch.model.NonLeafNode import NonLeafNode
from distance import Distance


class CfTree:
    def __init__(self):
        self.root = NonLeafNode()
        self.firstChild = self.root.entries[0]

    def insertCf(self, cf: ClusteringFeature):
        leaf = self.getClosestLeaf(cf, self.root)
        cf_node = leaf.getClosestCF(cf)

        if cf_node.canAbsorbCf(cf):
            cf_node.absorbCf(cf)
            self.updatePathSimple(leaf)
            return
        if leaf.canAddNewCf():
            leaf.addCF(cf)
            self.updatePathSimple(leaf)
        else:
            newLeaf = self.splitLeaf(leaf, cf)
            self.updatePathWithNewLeaf(newLeaf)

    def splitLeaf(self, leaf: LeafNode, cf: ClusteringFeature) -> LeafNode:
        leaf.cfs.append(cf)
        indices = Distance.getFurthest2Points(leaf.cfs)
        oldCf = [leaf.cfs[indices[0]]]
        newCf = [leaf.cfs[indices[1]]]

        for cf in leaf.cfs:
            if not cf is oldCf[0] and not cf is newCf[0]:
                if cf.calcualteDistance(oldCf[0]) < cf.calcualteDistance(newCf[0]):
                    oldCf.append(cf)
                else:
                    newCf.append(cf)

        index = leaf.parent.getChildIndex(leaf)
        leaf.cfs = oldCf
        leaf.parent.cfs[index] = leaf.sumAllCfs()

        newLeaf = LeafNode.LeafNode(newCf, parent=leaf.parent)
        leaf.next = newLeaf
        newLeaf.prev = newLeaf

        return newLeaf

    def updatePathSimple(self, child: LeafNode):
        parent = child.parent

        while parent is not None:
            idx = parent.getChildIndex(child)
            parent.cfs[idx] = child.sumAllCfs()
            child = parent
            parent = parent.parent

    def updatePathWithNewLeaf(self, newLeaf: LeafNode):
        # TODO: update the whole path in a loop
        if newLeaf.parent.canAddNode():
            newLeaf.parent.addNode(newLeaf)
        else:
            self.splitNonLeafNode(newLeaf)

    def splitNonLeafNode(self, node: CfNode):

        if node.parent != None:
            node.parent.addNode(node)
            nonLeafNode = node.parent
        else:
            nonLeafNode = node

        indices = Distance.getFurthest2Points(nonLeafNode.cfs)
        oldCf = [indices[0]]
        newCf = [indices[1]]
        nodeCfs = nonLeafNode.cfs
        nodeEntries = nonLeafNode.entries

        for idx, cf in enumerate(nonLeafNode.cfs):
            if not cf is nodeCfs[oldCf[0]] and not cf is nodeCfs[newCf[0]]:
                if cf.calcualteDistance(nodeCfs[oldCf[0]]) < cf.calcualteDistance(nodeCfs[newCf[0]]):
                    oldCf.append(idx)
                else:
                    newCf.append(idx)

        newNode = NonLeafNode()
        newNode.cfs = list(np.array(nodeCfs)[np.array(newCf)])
        newNode.entries = list(np.array(nodeEntries)[np.array(newCf)])

        for item in newNode.entries:
            item.parent = newNode

        nonLeafNode.cfs = list(np.array(nodeCfs)[np.array(oldCf)])
        nonLeafNode.entries = list(np.array(nodeEntries)[np.array(oldCf)])

        for item in nonLeafNode.entries:
            item.parent = nonLeafNode

        if nonLeafNode.parent is None:
            self.root = NonLeafNode()
            self.root.entries = []
            self.root.cfs = []
            self.root.addNode(nonLeafNode)
            self.root.addNode(newNode)
            print("new Height -> new root")
        else:
            if nonLeafNode.parent.canAddNode():
                print("add Node")
                nonLeafNode.parent.addNode(newNode)
            else:
                print("split again ")
                self.splitNonLeafNode(nonLeafNode.parent)

    def getClosestLeaf(self, cf: ClusteringFeature, nonLeafNode: NonLeafNode) -> LeafNode:
        cfNode = nonLeafNode.getClosestChild(cf)
        if cfNode is None:
            return None

        if cfNode.isLeaf:
            return cfNode
        else:
            return self.getClosestLeaf(cf, cfNode)

    def validate(self):
        self.validateNode(self.root)

    def validateNode(self, nonLeafNode: NonLeafNode) -> bool:
        n = 0
        # TODO: fix
        # for idx, node in enumerate(nonLeafNode.entries):
        #     n = self.calculateCfs(node)
        #     nNonLeaf = nonLeafNode.cfs[idx].N
        #     if n != nNonLeaf:
        #         print(False, idx)
        #         return False

        return True

    def calculateCfs(self, nonLeafNode: NonLeafNode) -> int:
        if nonLeafNode.isLeaf:
            return nonLeafNode.sumAllCfs().N
        else:
            n = 0
            for idx, node in enumerate(nonLeafNode.entries):
                n = self.validateNode(node)
                nNonLeaf = nonLeafNode.cfs[idx].N
                if n != nNonLeaf:
                    print(False, n, nNonLeaf)

    def getLeafList(self) -> list:
        next = self.firstChild
        leafs = [next]
        while next.next is not None:
            print("next")
            next = next.next
            leafs.append(next)

        return leafs

    def getLeafCfs(self) -> list:
        leafs = self.getLeafList()
        cfVectors = []
        for leaf in leafs:
            for cf in leaf.cfs:
                cfVectors.append(cf)
        return cfVectors

    def getVectorsFromCf(self, cfs: list) -> list:
        return [cf.getCenter() for cf in cfs]
