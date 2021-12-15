from typing import List

from pptree import print_tree

from flair.data import Sentence, Token


class NodeToken:
    def __init__(self, token: Token, tag_type: str):
        self.token: Token = token
        self.tag_type: str = tag_type
        self.children: List[NodeToken] = []

    def set_haed(self, parent):
        parent.children.append(self)

    def __str__(self):
        return " {}({}) ".format(self.token.text, self.token.get_labels(self.tag_type)[0].value)


def tree_printer(sentence: Sentence, tag_type: str):
    tree: List[NodeToken] = [NodeToken(token, tag_type) for token in sentence]
    for x in tree:
        if x.token.head_id != 0:
            head_token = x.token.get_head()

            for y in tree:
                if y.token == head_token:
                    x.set_haed(y)
        else:
            root_node = x
    print_tree(root_node, "children")
