import torch
import torch.nn.functional as F
from torch.nn import Linear

from torchvision.transforms import Compose, Resize, ToTensor

import flair
from flair.datasets import FeideggerCorpus, Corpus
from flair.embeddings import CharacterEmbeddings, BytePairEmbeddings, WordEmbeddings, FlairEmbeddings, MuseCrosslingualEmbeddings
from flair.embeddings import PrecomputedImageEmbeddings, IdentityImageEmbeddings, ConvTransformNetworkImageEmbeddings
from flair.embeddings import DocumentRNNEmbeddings, DocumentPoolEmbeddings
from flair.models.similarity_learning_model import SimilarityLearner
from flair.models.similarity_learning_model import SliceReshaper, ModelSimilarity, CosineSimilarity
from flair.models.similarity_learning_model import PairwiseBCELoss, RankingLoss
from flair.trainers import ModelTrainer
from flair.file_utils import cached_path

import pickle
import json
import os
import glob
import argparse
from pathlib import Path

optimizer_map = {'sgd':  torch.optim.SGD,
                 'adam': torch.optim.Adam}

word_embedding_map = {'char': CharacterEmbeddings,
                      'bpe': BytePairEmbeddings,
                      'wordstatic': WordEmbeddings,
                      'flair': FlairEmbeddings,
                      'muse': MuseCrosslingualEmbeddings}

document_embedding_map = {'pool': DocumentPoolEmbeddings,
                          'rnn': DocumentRNNEmbeddings}

mapping_map = {'linear': Linear,
               'net': ConvTransformNetworkImageEmbeddings}

similarity_map = {'model': ModelSimilarity,
                  'cosine': CosineSimilarity}

loss_map = {'bce': PairwiseBCELoss,
            'ranking': RankingLoss}

def run_experiment(parms):

    # text
    word_embeddings = []
    for word_embedding_name, word_embedding_parms in parms['word_embeddings']:
        word_embeddings += [word_embedding_map[word_embedding_name](**word_embedding_parms)]
    document_embedding = document_embedding_map[parms['document_embedding']['name']](word_embeddings, **parms['document_embedding']['parms'])

    dataset_home = parms['filenames']['dataset_home']

    # image
    if parms['image_input_type'] == 'images':
        transforms = Compose([Resize((367, 254), interpolation=3), ToTensor()]) # (275, 190)
        image_embedding = IdentityImageEmbeddings(transforms)
    elif parms['image_input_type'] == 'features':
        feature_type = parms['image_feature_type']
        image_features_filename = f'{dataset_home}/image_features/feidegger_{feature_type}_features.pickle'
        image_features = pickle.load(open(image_features_filename, 'rb'))
        json_link = 'https://raw.githubusercontent.com/zalandoresearch/feidegger/master/data/FEIDEGGER_release_1.1.json'
        json_local_path = cached_path(json_link, Path('datasets') / 'feidegger')
        dataset_data = json.load(open(json_local_path, 'r'))
        images_folder = os.path.join(os.path.dirname(json_local_path), 'images')
        url2embedding = {os.path.join(images_folder, os.path.basename(image_data['url'])): image_features[i] for i, image_data in enumerate(dataset_data)}
        image_embedding = PrecomputedImageEmbeddings(url2embedding, name=feature_type)

    corpus: Corpus = FeideggerCorpus(lowercase=True)

    print(corpus)

    # similarity
    target_mapping = None if parms['target_mapping'] is None else mapping_map[parms['target_mapping']['name']](**parms['target_mapping']['parms']).to(flair.device)
    # document: source, image: target
    if parms['source_mapping']['parms'] == 'image+1':
        # source => image  + 1, target = None (features)
        source_mapping_parms = (document_embedding.embedding_length, image_embedding.embedding_length+1)
    elif parms['source_mapping']['parms'] == 'image':
        # source => image, target = None (features)
        source_mapping_parms = (document_embedding.embedding_length, image_embedding.embedding_length)
    elif parms['source_mapping']['parms'] == 'target+1':
        # source => target + 1, target = MiniNet (images)
        source_mapping_parms = (document_embedding.embedding_length, target_mapping.embedding_length+1)
    elif parms['source_mapping']['parms'] == 'shared+1':
        # source => target + 1, target = Linear(feat_in, feat_shared)
        source_mapping_parms = (document_embedding.embedding_length, target_mapping.out_features+1)
    elif parms['source_mapping']['parms'] == 'shared':
        source_mapping_parms = (document_embedding.embedding_length, target_mapping.out_features)
    source_mapping = None if parms['source_mapping'] is None else mapping_map[parms['source_mapping']['name']](*source_mapping_parms).to(flair.device)
    if parms['similarity_measure']['name'] == 'model':
        similarity_measure_parms = ([(F.linear, {'weight': SliceReshaper(begin=0, end=source_mapping_parms[1]-1),
                                                'bias': SliceReshaper(begin=source_mapping_parms[1]-1)}),
                                    (torch.transpose, {'dim0': 1, 'dim1': 0})],)
    elif parms['similarity_measure']['name'] == 'order':
        similarity_measure_parms = parms['similarity_measure']['parms'] # TODO: fix the mess, either parms are dict or tuple
    else:
        similarity_measure_parms = {}
    similarity_measure = similarity_map[parms['similarity_measure']['name']](*similarity_measure_parms)
    similarity_loss = loss_map[parms['similarity_loss']['name']](**parms['similarity_loss']['parms'])

    similarity_model = SimilarityLearner(source_embeddings=document_embedding,
                                         target_embeddings=image_embedding,
                                         source_mapping=source_mapping,
                                         target_mapping=target_mapping,
                                         similarity_measure=similarity_measure,
                                         similarity_loss=similarity_loss,
                                         interleave_embedding_updates=parms['optim']['interleave_embedding_updates'])

    print(similarity_model)

    trainer: ModelTrainer = ModelTrainer(similarity_model, corpus, optimizer=optimizer_map[parms['optim']['optimizer_name']])

    output_folder = parms['filenames']['output_folder']

    trainer.train(
        f'{dataset_home}/{output_folder}',
        learning_rate=parms['optim']['lr'],
        mini_batch_size=parms['optim']['batch_size'],
        max_epochs=parms['optim']['max_epochs'],
        min_learning_rate=parms['optim']['min_lr'],
        anneal_factor=parms['optim']['anneal_factor'],
        patience=parms['optim']['patience'],
        shuffle=True,
        num_workers=6,
        embeddings_storage_mode='cpu',
        anneal_with_restarts=True,
        eval_on_train_fraction='dev'
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-path', metavar='S', type=str,
                        help='the path to the folder with experiment configurations to be run')
    parser.add_argument('--experiment-file', metavar='S', type=str,
                        help='the path to the file with the experiment configuation to be run')
    args = parser.parse_args()

    experiment_files = []
    if args.experiments_path is not None:
        experiment_files = list(glob.glob(os.path.join(args.experiments_path, '*.json')))

    if args.experiment_file is not None:
        experiment_files.append(args.experiment_file)

    for parm_filename in experiment_files:
        parms = json.load(open(parm_filename, 'r'))
        experiment_results_folder = os.path.join(parms['filenames']['dataset_home'], parms['filenames']['output_folder'])
        if os.path.exists(experiment_results_folder):
            print(f'Skipping experiment with parms:\n{parms}')
        else:
            print(f'Running experiment with parms:\n{parms}')
            run_experiment(parms)
