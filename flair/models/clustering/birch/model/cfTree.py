import logging

import numpy as np

from flair.models.clustering.birch.model.cfNode import CfNode
from flair.models.clustering.birch.model.clusteringFeature import ClusteringFeature
from flair.models.clustering.birch.model.leafNode import LeafNode
from flair.models.clustering.birch.model.nonLeafNode import NonLeafNode
from flair.models.clustering.distance import distance

log = logging.getLogger("flair")


class CfTree:
    def __init__(self):
        self.root = NonLeafNode()
        self.first_child = self.root.entries[0]

    def insert_cf(self, cf: ClusteringFeature):
        leaf = self.get_closest_leaf(cf, self.root)
        cf_entry = leaf.get_closest_cf(cf)

        if cf_entry.can_absorb_cf(cf):
            print("Absorb the cf")
            cf_entry.absorb_cf(cf)
            self.update_path_simple(leaf)
            return

        if leaf.can_add_new_cf():
            log.debug("Added cf to leaf")
            leaf.add_cf(cf)
            self.update_path_simple(leaf)
        else:
            log.debug("Split the leaf")
            new_leaf = self.split_leaf(leaf, cf)
            self.update_path_with_new_leaf(new_leaf)

    def split_leaf(self, leaf: LeafNode, cf: ClusteringFeature) -> LeafNode:
        leaf.cfs.append(cf)
        indices = distance.get_furthest_2_points(leaf.cfs)
        old_cf = [leaf.cfs[indices[0]]]
        new_cf = [leaf.cfs[indices[1]]]

        for cf in leaf.cfs:
            if not cf is old_cf[0] and not cf is new_cf[0]:
                if cf.calculate_distance(old_cf[0]) < cf.calculate_distance(new_cf[0]):
                    old_cf.append(cf)
                else:
                    new_cf.append(cf)

        index = leaf.parent.get_child_index(leaf)
        leaf.cfs = old_cf
        leaf.parent.cfs[index] = leaf.sum_all_cfs()

        new_leaf = LeafNode(new_cf, parent=leaf.parent)
        neighbor_leaf = leaf
        while neighbor_leaf.next is not None:
            neighbor_leaf = neighbor_leaf.next

        neighbor_leaf.next = new_leaf
        new_leaf.prev = neighbor_leaf

        return new_leaf

    def update_path_simple(self, child: LeafNode):
        parent = child.parent

        while parent is not None:
            idx = parent.get_child_index(child)
            parent.cfs[idx] = child.sum_all_cfs()
            child = parent
            parent = parent.parent

    def update_path_with_new_leaf(self, new_leaf: LeafNode):
        if new_leaf.parent.can_add_node():
            new_leaf.parent.add_node(new_leaf)
            self.update_path_simple(new_leaf)
        else:
            self.split_non_leaf_node(new_leaf)
            self.update_path_simple(new_leaf)

    def split_non_leaf_node(self, node: CfNode):

        if node.parent is not None:
            node.parent.add_node(node)
            non_leaf_node = node.parent
        else:
            non_leaf_node = node

        indices = distance.get_furthest_2_points(non_leaf_node.cfs)
        old_cf = [indices[0]]
        new_cf = [indices[1]]
        node_cfs = non_leaf_node.cfs
        node_entries = non_leaf_node.entries

        for idx, cf in enumerate(non_leaf_node.cfs):
            if not cf is node_cfs[old_cf[0]] and not cf is node_cfs[new_cf[0]]:
                if cf.calculate_distance(node_cfs[old_cf[0]]) < cf.calculate_distance(node_cfs[new_cf[0]]):
                    old_cf.append(idx)
                else:
                    new_cf.append(idx)

        new_node = NonLeafNode()
        new_node.cfs = list(np.array(node_cfs)[np.array(new_cf)])
        new_node.entries = list(np.array(node_entries)[np.array(new_cf)])

        for item in new_node.entries:
            item.parent = new_node

        non_leaf_node.cfs = list(np.array(node_cfs)[np.array(old_cf)])
        non_leaf_node.entries = list(np.array(node_entries)[np.array(old_cf)])

        for item in non_leaf_node.entries:
            item.parent = non_leaf_node

        if non_leaf_node.parent is None:
            self.root = NonLeafNode()
            self.root.entries = []
            self.root.cfs = []
            self.root.add_node(non_leaf_node)
            self.root.add_node(new_node)
            print("new Height -> new root")
        else:
            if non_leaf_node.parent.can_add_node():
                print("add Node")
                non_leaf_node.parent.add_node(new_node)
            else:
                print("split again ")
                self.split_non_leaf_node(non_leaf_node.parent)

    def validate(self, vectors: list):

        # check all the leafs
        cfs = self.get_leaf_cfs()
        test = [cf.N for cf in cfs]

        if sum(test) != len(vectors):
            log.error(
                "Error found N: " + str(sum(test)) + " entries in all Leafs but got: " + str(len(vectors)) + " vectors."
            )

    def get_closest_leaf(self, cf: ClusteringFeature, non_leaf_node: NonLeafNode) -> LeafNode:
        cf_node = non_leaf_node.get_closest_child(cf)
        if cf_node is None:
            return None

        if cf_node.is_leaf:
            return cf_node
        else:
            return self.get_closest_leaf(cf, cf_node)

    def get_leafs(self) -> list:
        next = self.first_child
        leafs = [next]
        while next.next is not None:
            next = next.next
            leafs.append(next)

        return leafs

    def get_leaf_cfs(self) -> list:
        leafs = self.get_leafs()
        cf_vectors = []
        for leaf in leafs:
            for cf in leaf.cfs:
                cf_vectors.append(cf)
        return cf_vectors

    def get_vectors_from_cf(self, cfs: list) -> list:
        return [cf.get_center() for cf in cfs]
