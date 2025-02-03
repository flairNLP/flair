import os
import numpy as np
import pandas as pd

categories_ids = ['4']
f_types = ['f05', 'f1']#, 'f2']
dataset='noise_crowd'

for category_id in categories_ids:
    optimize_F1s_output = open('optimal_category'+category_id+'_test_scores.csv','w')

    optimize_F1s_output.write('metric, f_score, modification, noise share, test score \n')

    for f_type in f_types:
        experiment_path = 'resources/relabel_cat'+category_id+'_ftype_'+f_type+os.sep+'category'+category_id

        for metric in os.listdir(experiment_path):
            print(os.path.join(experiment_path, metric, f_type))
            for modif in os.listdir(os.path.join(experiment_path, metric, f_type)):
                results_path = os.path.join(experiment_path, metric, f_type, modif)
                print(results_path)
                noise_shares = []
                for seed_path in os.listdir(results_path+os.sep+dataset):
                    fname = os.path.join(results_path, dataset, seed_path, 'noise_f1.txt')
                    if os.path.isfile(fname):
                        with open(fname) as f:
                            lines = f.read().strip().split('\n')
                            noise = lines[3].split(' ')[3]
                            noise_shares.append(float(noise))
                            print(noise_shares)
                test_results_fname = results_path+os.sep+dataset+os.sep+'test_results.tsv'
                if os.path.isfile(test_results_fname):
                    results_df = pd.read_csv(test_results_fname, delimiter='\t', header=0)
                    print(results_df[['mean']].values)
                    score = float(results_df[['mean']].values[0])
                    optimize_F1s_output.write(f"{''.join(metric.split('_')[1:])}, {f_type}, {modif}, {np.mean(noise_shares)}, {round(score, 4)}\n")
                    #optimize_F1s_output.write(f"{metric.split('_')[1:]}, {f_type}, {modif}, {0}, {0}\n")

