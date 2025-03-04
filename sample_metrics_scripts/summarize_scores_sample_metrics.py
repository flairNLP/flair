import os
import numpy as np
import pandas as pd

categories_ids = ['1','2','3','4']

def summarize_test_scores(results_tables_path, corpus_name):

    dataset = corpus_name

    for category_id in categories_ids:

        optimize_F1s_output = open(results_tables_path + os.sep + dataset+ os.sep+'optimal_category'+category_id+'_test_scores.csv','w')

        optimize_F1s_output.write('metric, f_score, modification, noise share, test score, std test score \n')

        experiment_path = results_tables_path+os.sep+'resources/relabel_cat'+category_id+os.sep+'category'+category_id

        for metric in os.listdir(experiment_path):

            for f_type in os.listdir(os.path.join(experiment_path,metric)):
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
                                noise = lines[0].split(' ')[0]
                                noise_shares.append(float(noise))
                                print(noise_shares)
                    test_results_fname = results_path+os.sep+dataset+os.sep+'test_results.tsv'
                    if os.path.isfile(test_results_fname):
                        results_df = pd.read_csv(test_results_fname, delimiter='\t', header=0)
                        print(results_df[['mean']].values)
                        score = float(results_df[['mean']].values[0])
                        stdev = float(results_df[['std']].values[0])
                        optimize_F1s_output.write(f"{'_'.join(metric.split('_')[1:])}, {f_type}, {modif}, {np.mean(noise_shares):.3f}, {round(score, 4):.3f}, {round(stdev, 4):.3f}\n")
                        #optimize_F1s_output.write(f"{metric.split('_')[1:]}, {f_type}, {modif}, {0}, {0}\n")