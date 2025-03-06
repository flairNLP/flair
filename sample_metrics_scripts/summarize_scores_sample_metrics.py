import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

categories_ids = ['1','2','3','4']


def summarize_test_scores(results_tables_path, corpus_name):

    dataset = corpus_name

    for category_id in categories_ids:
        
        test_scores_tables_path = results_tables_path + os.sep + dataset+ os.sep +'test_scores'

        if not os.path.exists(test_scores_tables_path):
            os.makedirs(test_scores_tables_path)

        optimize_F1s_output = open(test_scores_tables_path + os.sep+'category'+category_id+'_test_scores.csv','w')

        optimize_F1s_output.write('metric, f_score, modification, noise share, test score, std test score \n')

        experiment_path = results_tables_path+os.sep+'resources/relabel_cat'+category_id+os.sep+'category'+category_id

        for metric in os.listdir(experiment_path):
            # this includes both standard and EE metrics
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


def merge_tables(results_tables_path, modes):

    for category in categories_ids:

        data2 = pd.read_csv(results_tables_path+os.sep+'test_scores'+os.sep+'category'+category+'_test_scores.csv',header = 0, index_col=[0,1, 2])

        full_data = None
        for mode in modes:

            base_path1 = f'{results_tables_path}/{mode}_mode'

            data1 = pd.read_csv(base_path1+os.sep+'optimal_F1s_category'+category+'.csv',header = 0, index_col=[0,1])
            print(data1)

            if full_data is None:
                full_data = pd.merge(data1, data2, left_index=True, right_index=True)
            else:
                full_data = pd.concat([full_data, pd.merge(data1, data2, left_index=True, right_index=True)], axis = 0)

        final_tables_path = results_tables_path+os.sep+'final_tables'

        if not os.path.exists(final_tables_path):
            os.makedirs(final_tables_path)

        full_data.to_csv(final_tables_path+os.sep+f'category{category}_merged_optimal_table.csv',index=True,header=True, float_format='%.3f')


def plot_metric_distributions(base_path, seeds, mode, sample_metrics, dset = 'train', y_limit=2000, max_epochs=11):
    '''
    Plots the distribution of PD values, for different categories of samples and for each epoch.

    For only one seed, the first one (100).
    Categories:

    1. correct prediction and predicted label is O 
    2. correct prediction and predicted label is not O 
    3. incorrect prediction and predicted label is O 
    4. incorrect prediction and predicted label is not O 

    The four categories are shown side by side. 

    The histograms for noisy and clean samples are shown in different colors.

    In addition, all memorized samples are printed out in a file: memorized_samples_with_memorization_epoch.csv
    , where the memorization epoch is the epoch where the noisy label is predicted and after which the prediction doesn't change.
    '''

    if mode == 'EE':
        exp_paths = [f'{seed}_with_init-0.3/' for seed in seeds]
    else:
        exp_paths = [f'{seed}/' for seed in seeds]

    flag_name = 'Correct prediction flag'
    secondary_flag_name = 'Noise flag'

    plt.legend(markerscale=2)

    filepath = base_path + exp_paths[0]+'epoch_log'+'_0.log'

    if not os.path.exists(filepath):
        start_index = 1
    else:
        start_index = 0

    if dset == 'train':
        ext = ''
    else:
        ext = '_'+dset
    
    for metric in sample_metrics:
        for i in [str(i) for i in range(start_index, max_epochs)]:

            for exp_path in exp_paths: 
                ## plot metric distribution for only one seed
                
                path = base_path + exp_path

                filepath = path+'epoch_log'+'_'+i+ext+'.log'
                print(filepath)
                if not os.path.exists(path+'histograms_'+metric):
                    os.mkdir(path+'histograms_'+metric)

                try:
                    df = pd.read_csv(filepath,  delimiter='\t', header=0, quoting=csv.QUOTE_NONE)
                except:
                    continue
                print(df.columns)
                df[flag_name] = df['predicted'] == df['noisy']

                df.rename(columns = {'noisy_flag':secondary_flag_name}, inplace=True)

                print(df.groupby(secondary_flag_name).count())
                print(len(df))
                if metric in ['pd','fl','tal','tac','mild','mild_f','mild_m']:
                    max_metric = df[metric].max()
                    binwidth = 1
                    if metric == 'mild':
                        binrange = (-max_metric, max_metric)
                        print(binrange)
                    else:
                        binrange = (0, max_metric)
                elif metric in ['le','cross_entropy','entropy']:
                    max_metric = 8
                    binwidth = 0.1
                    binrange = (0,max_metric)
                elif metric == 'pehist':
                    max_metric = 0.1
                    binwidth = 0.005
                    binrange = (0,max_metric)
                else:
                    max_metric = df[metric].max()
                    binwidth = 0.1
                    binrange = (0,1)

                df.rename(columns = {'noisy':'label'}, inplace=True)

                fig, axes = plt.subplots(2, 2, figsize=(14, 4))

                #cat1
                sns.histplot(df[(df[flag_name] == True)  & (df['label']=='O')] , x=metric, binwidth=binwidth, binrange=binrange, ax = axes[0,0], hue= secondary_flag_name)
                axes[0,0].set_title('Epoch '+i+' - Correct prediction (observed label O)')
                axes[0,0].set_xlim(binrange)
                axes[0,0].set_ylim((0,y_limit))

                #cat2
                sns.histplot(df[(df[flag_name] == True) & (df['label'] != 'O') ] , x=metric, binwidth=binwidth, binrange=binrange, ax = axes[1,0], hue = secondary_flag_name)
                axes[1,0].set_title('Epoch '+i+' - Correct prediction (observed label non-O)')
                axes[1,0].set_xlim(binrange)
                axes[1,0].set_ylim((0,y_limit))
                
                #cat3
                sns.histplot(df[(df[flag_name] == False) & (df['label']=='O')] , x=metric, binwidth=binwidth, binrange=binrange, ax = axes[0,1], hue = secondary_flag_name)
                axes[0,1].set_title('Epoch '+i+' - Incorrect prediction (observed label O)')
                axes[0,1].set_xlim(binrange)
                axes[0,1].set_ylim((0,y_limit))

                #cat4
                sns.histplot(df[(df[flag_name] == False) & (df['label'] != 'O') ] , x=metric, binwidth=binwidth, binrange=binrange, ax = axes[1,1], hue = secondary_flag_name)
                axes[1,1].set_title('Epoch '+i+' - Incorrect prediction (observed label non O)')
                axes[1,1].set_xlim(binrange)
                axes[1,1].set_ylim((0,y_limit))

                fig.tight_layout() 
                fig.savefig(path+'histograms_'+metric+os.sep+metric+'_distribution_epoch_'+i+ext+'.png')
                plt.close()
