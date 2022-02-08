import unittest

import torch

from flair.data import Sentence
# from unittest.mock import MagicMock, patch

from flair.embeddings import (
    GazetteerEmbeddings,
)


class GazetteerEmbeddingsTest(unittest.TestCase):

    def setUp(self) -> None:
        label_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                      'I-MISC': 8}
        self.gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                            label_dict=label_dict)

    def test_matching_methods(self):
        self.assertEqual(self.gazetteer_embedding.matching_methods, ['full_match', 'partial_match'])

    def test_embedding(self):
        sentence = Sentence('I Love Paris .')
        self.gazetteer_embedding.embed(sentence)
        for token in sentence.tokens:
            self.assertEqual(len(token.get_embedding()), 1)

            token.clear_embeddings()

            self.assertEqual(len(token.get_embedding()), 0)
