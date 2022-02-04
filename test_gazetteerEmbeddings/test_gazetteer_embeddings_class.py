import unittest

import torch

from flair.data import Sentence
# from unittest.mock import MagicMock, patch

from flair.embeddings import (
    GazetteerEmbeddings,
)


class GazetteerEmbeddingsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(gazetteers=["test"])

    def test_matching_methods(self):
        self.assertEqual(self.gazetteer_embedding.matching_methods, ['full_match', 'partial_match'])

    def test_embedding(self):
        sentence = Sentence('I Love Paris .')
        self.gazetteer_embedding.embed(sentence)
        for token in sentence.tokens:
            self.assertEqual(len(token.get_embedding()), 0)

            token.clear_embeddings()

            self.assertEqual(len(token.get_embedding()), 0)
