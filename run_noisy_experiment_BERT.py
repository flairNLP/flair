import flair.datasets
import flair, torch
from flair.embeddings import TransformerEmbeddings

from flair.trainers import ModelTrainer
from flair.models import TextClassifierLossModifications
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

import random
import os
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-v", "--variant", help="pe_norm or cce", default='cce')
argParser.add_argument("-d", "--dataset", help="trec is the only option for now", default='trec')
# for now only downsampled yahoo answers
argParser.add_argument("-g", "--gpu", help="gpu id", default=0)
argParser.add_argument("-s", "--num_seeds", help="number of seeds", default=1, type=int)
argParser.add_argument("-cdn", "--class_dependent_noise", help="number of seeds", default=True, type=bool)
argParser.add_argument("-n", "--noise_share", help="number between 0 and 90", default=0.25)
argParser.add_argument("-e", "--exp_name", help="experiment name / results base path", default='test_metrics')

argParser.add_argument('--fix_split_seed', dest='fix_split_seed_flag',  help="add argument if validation split and downsampling seed should be fixed", action='store_true')
argParser.set_defaults(fix_split_seed_flag=False)

## Add LR and BS as arguments
lr = 5e-6
batch_size = 16
num_epochs = 10

args = argParser.parse_args()

datasets_label_order_ntm = {'trec': ['ABBR','ENTY','DESC','HUM','LOC','NUM'],'yahoo_answers':[''] }
datasets_label_type = {'trec': 'question_class','yahoo_answers':'question_type' }

if args.num_seeds == 6:
    seeds = [42,52,62,5000,8989,444333]
elif args.num_seeds==1:
    seeds = [42]
else:
    seeds = random.sample(range(1, 2**32), args.num_seeds)


flair.device = torch.device('cuda:'+str(args.gpu)) 

experiment_path = args.exp_name+'_'+args.dataset+'_'+args.variant+'_'+str(args.class_dependent_noise)+'_'+str(args.noise_share)
results_path = 'results'+os.sep+experiment_path
base_resources_path = 'resources'+os.sep+'noisy_classification'

label_type = datasets_label_type[args.dataset]
                       
if not os.path.isdir(results_path):
    os.mkdir(results_path)

outfile = open(results_path+os.sep+"scores.log", 'w')
outfile.write(f"train_acc, val_acc, test_acc\n")

test_accs=[]
val_accs=[]
train_accs=[]

for seed in seeds:
    experiment_path_with_seed = args.exp_name+'_'+args.dataset+'_'+args.variant+'_'+str(args.class_dependent_noise)+'_'+str(args.noise_share)+'_'+str(seed)
    resources_path = base_resources_path+os.sep+experiment_path_with_seed

    flair.set_seed(seed)

    random_split_seed = 42 if args.fix_split_seed else seed
    if args.dataset == 'trec':
        corpus = flair.datasets.TREC_6(split_seed=random_split_seed)
    else:
        corpus = flair.datasets.YAHOO_ANSWERS(split_seed=random_split_seed).downsample(0.1)
    
    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
    labels = label_dict.get_items()

    if args.class_dependent_noise == True:
        nt_matrix = np.load(args.dataset+'_ntm.npy')
        ntm_dict = {}
        for i,label in enumerate(datasets_label_order_ntm[args.dataset]):
            ntm_dict[label] = nt_matrix[i]
        share, new_ntm = corpus.add_label_noise(noise_share=args.noise_share, labels=datasets_label_order_ntm[args.dataset], label_type=label_type, noise_transition_matrix=ntm_dict) # seed removed from corrupt_labels
        np.savetxt(results_path+os.sep+'CDN_matrix_'+str(seed)+'.csv', new_ntm, fmt="%d", delimiter=",")
    else:
        share = corpus.add_label_noise(noise_share=args.noise_share, labels=datasets_label_order_ntm[args.dataset], label_type=label_type)

    embeddings =TransformerEmbeddings('bert-base-cased', fine_tune=True, layers='all')
    classifier = TextClassifierLossModifications(embeddings, label_dictionary=label_dict, label_type=label_type, batch_avg=False, loss=args.variant, calculate_sample_metrics=True)
    trainer = ModelTrainer(classifier, corpus)

    trainer.fine_tune(resources_path,learning_rate=lr,mini_batch_size=batch_size,max_epochs=num_epochs, scheduler=OneCycleLR, cycle_momentum=True)

    corpus.print_noisy_dataset(label_type=label_type,path=resources_path)
    
    layer_eval_test = classifier.evaluate(corpus.test, gold_label_type=label_type)
    layer_eval_dev = classifier.evaluate(corpus.dev, gold_label_type=label_type)
    layer_eval_train = classifier.evaluate(corpus.train, gold_label_type=label_type)
    outfile.write(f"{str(layer_eval_train.main_score)}")
    outfile.write(f", {str(layer_eval_dev.main_score)}")
    outfile.write(f", {str(layer_eval_test.main_score)}\n")
    train_accs.append(layer_eval_train.main_score)
    val_accs.append(layer_eval_dev.main_score)
    test_accs.append(layer_eval_test.main_score)

outfile.write(f"{str(sum(train_accs) / len(train_accs))}")
outfile.write(f", {str(sum(val_accs) / len(val_accs))}")
outfile.write(f", {str(sum(test_accs) / len(test_accs))}\n")
outfile.close()
