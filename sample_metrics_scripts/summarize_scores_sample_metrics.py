import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from optimize_metric_parameters import *
import logging

logger_experiment = logging.getLogger(__name__)
logger_experiment.setLevel(level="INFO")


CATEGORIES = [
    { 
        'id':'1',
        'name':'Correct prediction (observed label O)',
        'axes_indices':(0,0),
        'correct_prediction_flag':True,
        'observed_label':'O',
        'color':'#55a868'
    },
    {
        'id': '2',
        'name': 'Incorrect prediction (observed label O)',
        'axes_indices':(0,1),
        'correct_prediction_flag':False,
        'observed_label':'O',
        'color':'#c44e52'
    },                
    {
        'id':'3',
        'name':'Correct prediction (observed label non-O)',
        'axes_indices':(1,0),
        'correct_prediction_flag':True,
        'observed_label':'non-O',
        'color':'#55a868'
    },

    {
        'id':'4',
        'name':'Incorrect prediction (observed label non O)',
        'axes_indices':(1,1),
        'correct_prediction_flag':False,
        'observed_label':'non-O',
        'color':'#c44e52'
    }
]

CORRECT_PREDICTION_FLAG_NAME = 'correct_prediction_flag'
NOISE_FLAG_NAME = 'noisy_flag'

def summarize_test_scores(results_tables_path, source_corpus, corpus_name, resources_path,   categories_ids, merged_parameters = True):
    '''
    Example path to test scores: source_noise_crowd_target_noise_llm/test_scores
    '''
    dataset = corpus_name

    for category_id in categories_ids:
        
        test_scores_tables_path = f"{results_tables_path}/source_{source_corpus}_target_{corpus_name}/test_scores"

        if not os.path.exists(test_scores_tables_path):
            os.makedirs(test_scores_tables_path)

        optimize_F1s_output = open(test_scores_tables_path + os.sep+'category'+category_id+'_test_scores.csv','w')

        optimize_F1s_output.write('metric, f_score, modification, noise share, test score, std test score \n')

        experiment_path = f"{resources_path}/relabel_cat{category_id}_source_{source_corpus}/category{category_id}"

        for metric in os.listdir(experiment_path):
            # this includes both standard and EE metrics
            for f_type in os.listdir(os.path.join(experiment_path,metric)):
                for modif in os.listdir(os.path.join(experiment_path, metric, f_type)):

                    results_path = os.path.join(experiment_path, metric, f_type, modif)
                    if os.path.exists(results_path+os.sep+dataset):

                        logger_experiment.debug(results_path)
                        noise_shares = []
                    
                        for seed_path in os.listdir(results_path+os.sep+dataset):
                            fname = os.path.join(results_path, dataset, seed_path, 'noise_f1.txt')
                            if os.path.isfile(fname):
                                with open(fname) as f:
                                    lines = f.read().strip().split('\n')
                                    noise = lines[0].split(' ')[0]
                                    noise_shares.append(float(noise))
                                    logger_experiment.debug(noise_shares)
                        test_results_fname = results_path+os.sep+dataset+os.sep+'test_results.tsv'
                        if os.path.isfile(test_results_fname):
                            results_df = pd.read_csv(test_results_fname, delimiter='\t', header=0)
                            logger_experiment.debug(results_df[['mean']].values)
                            score = float(results_df[['mean']].values[0])
                            stdev = float(results_df[['std']].values[0])
                            optimize_F1s_output.write(f"{'_'.join(metric.split('_')[1:])}, {f_type}, {modif}, {np.mean(noise_shares):.3f}, {round(score, 4):.3f}, {round(stdev, 4):.3f}\n")
                            #optimize_F1s_output.write(f"{metric.split('_')[1:]}, {f_type}, {modif}, {0}, {0}\n")


def merge_tables(results_tables_path, source_corpus, corpus_name, modes, categories_ids, merged_parameters = True):

    for category in categories_ids:

        test_scores_df = pd.read_csv(f"{results_tables_path}/source_{source_corpus}_target_{corpus_name}/test_scores/category{category}_test_scores.csv",header = 0, index_col=[0,1, 2])

        full_data = None
        for mode in modes:
            parameter_settings_tables_path = f"{results_tables_path}/{source_corpus}/{mode}_mode"
            parameter_settings_df = pd.read_csv(f"{parameter_settings_tables_path}/optimal_F1s_category{category}.csv",header = 0, index_col=[0,1])
            logger_experiment.debug(parameter_settings_df)

            if full_data is None:
                full_data = pd.merge(parameter_settings_df, test_scores_df, left_index=True, right_index=True)
            else:
                full_data = pd.concat([full_data, pd.merge(parameter_settings_df, test_scores_df, left_index=True, right_index=True)], axis = 0)

        final_tables_path = f"{results_tables_path}/source_{source_corpus}_target_{corpus_name}/final_tables"

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

    '''

    if mode == 'EE':
        exp_paths = [f'{seed}_with_init-0.3/' for seed in seeds]
    else:
        exp_paths = [f'{seed}/' for seed in seeds]

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
                logger_experiment.debug(filepath)
                if not os.path.exists(path+'histograms_'+metric):
                    os.mkdir(path+'histograms_'+metric)

                try:
                    df = pd.read_csv(filepath,  delimiter='\t', header=0, quoting=csv.QUOTE_NONE)
                except:
                    continue
                logger_experiment.debug(df.columns)
                df[CORRECT_PREDICTION_FLAG_NAME] = df['predicted'] == df['noisy']

                logger_experiment.debug(df.groupby(NOISE_FLAG_NAME).count())
                logger_experiment.debug(len(df))
                if metric in ['pd','fl','tal','tac','mild','mild_f','mild_m']:
                    max_metric = df[metric].max()
                    binwidth = 1
                    if metric == 'mild':
                        binrange = (-max_metric, max_metric)
                        logger_experiment.debug(binrange)
                    else:
                        binrange = (0, max_metric)
                elif metric in ['le','cross_entropy','entropy','pehist']:
                    max_metric = 8
                    binwidth = 0.1
                    binrange = (0,max_metric)
                else:
                    max_metric = df[metric].max()
                    binwidth = 0.1
                    binrange = (0,1)

                df.rename(columns = {'noisy':'label'}, inplace=True)

                fig, axes = plt.subplots(2, 2, figsize=(14, 4))
                for category in CATEGORIES:
                    #cat1
                    if category['observed_label'] == 'O':
                        dataset = df[(df[CORRECT_PREDICTION_FLAG_NAME] == category['correct_prediction_flag']) & (df['label'] == 'O') ]
                    else:
                        dataset = df[(df[CORRECT_PREDICTION_FLAG_NAME] == category['correct_prediction_flag']) & (df['label'] != 'O') ]
                    sns.histplot( dataset, x=metric, binwidth=binwidth, binrange=binrange, ax = axes[category['axes_indices']], hue= NOISE_FLAG_NAME)
                    axes[category['axes_indices']].set_title('Epoch '+i+' - '+category['name'])
                    axes[category['axes_indices']].set_xlim(binrange)
                    axes[category['axes_indices']].set_ylim((0,y_limit))

                fig.tight_layout() 
                fig.savefig(path+'histograms_'+metric+os.sep+metric+'_distribution_epoch_'+i+ext+'.png')
                plt.close()


def plot_category_membership_through_epochs(base_paths, corpus_name, seeds, dset = 'train', max_epochs=11):

    exp_paths={}
    exp_paths['EE'] = [f'{seed}_with_init-0.3/' for seed in seeds]
    exp_paths['standard'] = [f'{seed}/' for seed in seeds]

    plt.legend(markerscale=2)
    plt.style.use('seaborn-v0_8-whitegrid')
    seaborn_blue = '#4C72B0'
    seaborn_orange = '#DD8452'
    categories_counts = {}

    for mode in ['EE','standard']:
        logger_experiment.debug(mode)
        filepath = f"{base_paths[mode]}/{corpus_name}/{exp_paths[mode][0]}/epoch_log_0.log"

        if not os.path.exists(filepath):
            start_index = 1
        else:
            start_index = 0

        if dset == 'train':
            ext = ''
        else:
            ext = '_'+dset

        # for each mode, category and sample type (clean or noisy): save a list with sample counts from all epochs
        categories_counts[mode] = {cat['id']: {'clean':[], 'noisy':[]} for cat in CATEGORIES} 

        for i in [str(i) for i in range(start_index, max_epochs)]:

            for exp_path in exp_paths[mode][:1]: 
                ## temporary: plot for only one seed - the first one
                
                path = f"{base_paths[mode]}/{corpus_name}/{exp_paths[mode][0]}/"
                filepath = path+'epoch_log'+'_'+i+ext+'.log'
                epoch_log_df = pd.read_csv(filepath,  delimiter='\t', header=0, quoting=csv.QUOTE_NONE)

                epoch_log_df[CORRECT_PREDICTION_FLAG_NAME] = epoch_log_df['predicted'] == epoch_log_df['noisy']

                for category in CATEGORIES:
                    category_id = category['id']

                    # take the corresponding data subset
                    if category['observed_label'] == 'O':
                        category_epoch_log_df = epoch_log_df[(epoch_log_df[CORRECT_PREDICTION_FLAG_NAME] == category['correct_prediction_flag'])  & (epoch_log_df['noisy']=='O')]
                    else:
                        category_epoch_log_df = epoch_log_df[(epoch_log_df[CORRECT_PREDICTION_FLAG_NAME] == category['correct_prediction_flag'])  & (epoch_log_df['noisy']!='O')]

                    # count clean and noisy samples
                    count_clean, count_noisy = len(category_epoch_log_df[NOISE_FLAG_NAME].values) - sum(category_epoch_log_df[NOISE_FLAG_NAME].values), sum(category_epoch_log_df[NOISE_FLAG_NAME].values)

                    categories_counts[mode][category_id]['clean'].append(count_clean)
                    categories_counts[mode][category_id]['noisy'].append(count_noisy)            

    # Plot line plots with number of samples per-category. 
    # y-axis: number of samples
    # x-acis: epoch number
    # dashed lines: EE runs
    # solid lines: standard runs
    # clean and noisy samples are in different colors
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    for category in CATEGORIES:
        category_id = category['id']
        axes[category['axes_indices']].plot(categories_counts['EE'][category_id]['clean'], color=seaborn_blue, linestyle='dashed', marker='o', label= 'EE: clean')
        axes[category['axes_indices']].plot(categories_counts['EE'][category_id]['noisy'], color=seaborn_orange, linestyle='dashed', marker='o',label= 'EE: noisy')
        axes[category['axes_indices']].plot(categories_counts['standard'][category_id]['clean'], color=seaborn_blue, linestyle='solid', marker='o', label= 'standard: clean')
        axes[category['axes_indices']].plot(categories_counts['standard'][category_id]['noisy'], color=seaborn_orange, linestyle='solid', marker='o', label= 'standard: noisy')
        axes[category['axes_indices']].set_title(category['name'])
    axes[1,1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{base_paths['standard']}/{corpus_name}/lineplots_category_memberships_{corpus_name}.png")
    plt.close()


    # Plot barplots of numbers of samples; per (observed) label type
    # 4 stacked barplots (stack number of samples in pairs of complementary categories)
    # (observed label O is left: category 1 or 2)
    # (observed label non-O is right: category 3 or 4)
    # x-axis: epoch number
    # training modes (standard or EE) have bars with different opacity
    # categories have different colors

    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    width = 0.25

    bottom = np.zeros(10)
    for cat1 in [CATEGORIES[0], CATEGORIES[1]]:
        
        p = axes[0,0].bar(x= range(10), height = categories_counts['standard'][cat1['id']]['clean'][:10] , width=width, color=cat1['color'], label='standard: category'+cat1['id'], bottom=bottom)
        bottom += categories_counts['standard'][cat1['id']]['clean'][:10] 
    bottom = np.zeros(10)
    for cat1 in [CATEGORIES[0], CATEGORIES[1]]:
        p = axes[0,0].bar(x= [c + width for c in range(10)], height = categories_counts['EE'][cat1['id']]['clean'][:10] , width=width, color=cat1['color'], label='EE: category'+cat1['id'], alpha=0.5, bottom=bottom)
        bottom += categories_counts['EE'][cat1['id']]['clean'][:10] 
    axes[0,0].set_title('Category 1/2 (Observed label O) - clean')

    bottom = np.zeros(10)
    for cat1 in [CATEGORIES[0], CATEGORIES[1]]:
        p = axes[1,0].bar(x= range(10), height = categories_counts['standard'][cat1['id']]['noisy'][:10] , width=width, color=cat1['color'], label='standard: category'+cat1['id'], bottom=bottom)
        bottom += categories_counts['standard'][cat1['id']]['noisy'][:10] 
    bottom = np.zeros(10)
    for cat1 in [CATEGORIES[0], CATEGORIES[1]]:
        p = axes[1,0].bar(x= [c + width for c in range(10)], height = categories_counts['EE'][cat1['id']]['noisy'][:10]  , width=width, color=cat1['color'], label='EE: category'+cat1['id'], alpha=0.5, bottom=bottom)
        bottom += categories_counts['EE'][cat1['id']]['noisy'][:10] 
    axes[1,0].set_title('Category 1/2 (Observed label O) - noisy')


    bottom = np.zeros(10)
    for cat1 in [CATEGORIES[2], CATEGORIES[3]]:
        p = axes[0,1].bar(x= range(10), height = categories_counts['standard'][cat1['id']]['clean'][:10] ,   width=width, color=cat1['color'], label='standard: category'+cat1['id'], bottom=bottom)
        bottom += categories_counts['standard'][cat1['id']]['clean'][:10] 
    bottom = np.zeros(10)
    for cat1 in [CATEGORIES[2], CATEGORIES[3]]:
        p = axes[0,1].bar(x= [c + width for c in range(10)], height = categories_counts['EE'][cat1['id']]['clean'][:10],   width=width, color=cat1['color'], label='EE: category'+cat1['id'], alpha=0.5, bottom=bottom)
        bottom += categories_counts['EE'][cat1['id']]['clean'][:10] 
    axes[0,1].set_title('Category 3/4 (Observed label non-O) - clean')


    bottom = np.zeros(10)
    for cat1 in [CATEGORIES[2], CATEGORIES[3]]:
        p = axes[1,1].bar(x= range(10), height = categories_counts['standard'][cat1['id']]['noisy'][:10] ,   width=width, color=cat1['color'], label='standard: category'+cat1['id'], bottom=bottom)
        bottom += categories_counts['standard'][cat1['id']]['noisy'][:10] 
    bottom = np.zeros(10)
    for cat1 in [CATEGORIES[2], CATEGORIES[3]]:
        p = axes[1,1].bar(x= [c + width for c in range(10)], height = categories_counts['EE'][cat1['id']]['noisy'][:10] ,   width=width, color=cat1['color'], label='EE: category'+cat1['id'], alpha=0.5, bottom=bottom)
        bottom += categories_counts['EE'][cat1['id']]['noisy'][:10] 
    axes[1,1].set_title('Category 3/4 (Observed label non-O) - noisy')

    handles, labels = axes[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol = 4, bbox_to_anchor = (0.5, 0),)
    fig.savefig(f"{base_paths['standard']}/{corpus_name}/lineplots_category_memberships_version2_{corpus_name}.png")
    plt.close()