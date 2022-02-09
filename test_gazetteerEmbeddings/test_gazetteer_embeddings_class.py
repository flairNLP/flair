import os
import unittest
import os
from unittest import mock

from flair.data import Sentence
from unittest.mock import patch

from flair.embeddings import (
    GazetteerEmbeddings,
)


class GazetteerEmbeddingsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.label_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                           'I-MISC': 8}

    def test_matching_methods(self):
        with patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=self.label_dict,
                                                                           full_mathing=True,
                                                                           partial_matching=False)
            self.assertEqual(gazetteer_embedding.matching_methods, ['full_match'])

    def test_process_gazetteers(self):
        gazetteer_files = ['eng-ORG-alias-wd.txt', 'eng-LOC-alias-wd.txt']
        with patch.object(os, 'listdir', return_value=gazetteer_files):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=self.label_dict)
            self.assertEqual(gazetteer_embedding.gazetteer_file_list, gazetteer_files)

    def test_embedding(self):
        with mock.patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=self.label_dict)
            sentence = Sentence('I Love Paris .')
            gazetteer_embedding.embed(sentence)
            for token in sentence.tokens:
                self.assertEqual(len(token.get_embedding()), 1)

                token.clear_embeddings()

                self.assertEqual(len(token.get_embedding()), 0)
