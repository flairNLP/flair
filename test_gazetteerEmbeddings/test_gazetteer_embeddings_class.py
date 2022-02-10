import unittest
import os
from flair.data import Sentence
from unittest.mock import patch, MagicMock, call, mock_open

from flair.embeddings import (
    GazetteerEmbeddings,
)


class GazetteerEmbeddingsTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_matching_methods(self):
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_mathing=True,
                                                                           partial_matching=False)
            self.assertEqual(gazetteer_embedding.matching_methods, ['full_match'])

    def test_get_gazetteers_good1(self):
        label_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                      'I-MISC': 8}
        gazetteer_files = ['eng-ORG-alias-wd.txt', 'eng-LOC-alias-wd.txt']
        with patch.object(os, 'listdir', return_value=gazetteer_files), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=label_dict)
            self.assertEqual(gazetteer_embedding.gazetteer_file_list, gazetteer_files)

    def test_get_gazetteers_good2(self):
        label_dict = {'O': 0, 'PER': 1, 'ORG': 2, 'LOC': 3, 'MISC': 4}
        gazetteer_files = ['eng-ORG-alias-wd.txt', 'eng-LOC-alias-wd.txt']
        with patch.object(os, 'listdir', return_value=gazetteer_files), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=label_dict)
            self.assertEqual(gazetteer_embedding.gazetteer_file_list, gazetteer_files)

    def test_process_gazetteers_good(self):
        label_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                      'I-MISC': 8}
        gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources",
                                                                       label_dict=label_dict)
        self.assertEqual(len(gazetteer_embedding.gazetteers_dicts), 2)
        self.assertEqual(len(gazetteer_embedding.gazetteers_dicts[0]), 13)
        self.assertEqual(len(gazetteer_embedding.gazetteers_dicts[1]), 6)
