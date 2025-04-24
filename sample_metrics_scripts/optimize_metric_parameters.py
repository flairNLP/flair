
import os
import csv
import numpy as np
import pandas as pd
from sklearn import metrics
from collections import Counter
import json


'''
Categories:

1. correct prediction and predicted label is O - (0,0)
2. correct prediction and predicted label is not O - (1,0)
3. incorrect prediction and predicted label is O - (0,1)
4. incorrect prediction and predicted label is not O - (1,1) 
'''

F_SCORE_NAMES = ['f05','f1','f2']

CATEGORIES = [
    { 
        'id':'1',
        'name':'Correct prediction (observed label O)',
        'axes_indices':(0,0),
        'correct_prediction_flag':True,
        'observed_label':'O'
    },
    {
        'id': '2',
        'name': 'Incorrect prediction (observed label O)',
        'axes_indices':(0,1),
        'correct_prediction_flag':False,
        'observed_label':'O'
    },                
    {
        'id':'3',
        'name':'Correct prediction (observed label non-O)',
        'axes_indices':(1,0),
        'correct_prediction_flag':True,
        'observed_label':'non-O'
    },

    {
        'id':'4',
        'name':'Incorrect prediction (observed label non O)',
        'axes_indices':(1,1),
        'correct_prediction_flag':False,
        'observed_label':'non-O'
    }
]

def get_metrics_thresholds(y_test,  y_pred_proba, metric, direction, epoch, total_num_noisy):
    # set the list of thresholds based on the metric

    if metric in ['msp','BvSB','confidence', 'correctness','iter_norm']:
        thresholds = np.arange(0, 1, 0.1)
    elif metric in ['variability']:
        thresholds = np.arange(0, 0.5, 0.05)
    elif metric in [ 'tac', 'pd', 'fl', 'tal' ]:
        thresholds = np.arange(0, 25, 1)
    elif metric in [ 'cross_entropy', 'entropy', 'le', 'pehist' ]:
        minimum = 0
        maximum = max(y_pred_proba) # this can cause a different number of thresholds among seeds
        if metric != 'pehist':
            thresholds = np.arange(minimum, maximum, 0.1)
        else:
            thresholds = np.arange(minimum, maximum, 0.05)
    elif metric == 'mild':
        thresholds = np.arange(-epoch, epoch, 1)
    elif metric == 'mild_f':
        thresholds = np.arange(1, epoch +1 , 1)
    else:
        thresholds = np.arange(1, epoch+1, 1)

    # for each threshold, calculate precision and recall scores
    precisions, recalls = [], []
    for th in thresholds:
        
        # set y_pred based on the direction
        # y_pred = 1 if sample is predicted noisy
        # y_pred = 0 is sample is predicted clean
        if direction == 'left':
            y_pred = np.where(y_pred_proba < th, 1, 0)
        else:
            y_pred = np.where(y_pred_proba > th, 1, 0)
        
        # the precision score is standard precision (for the noisy class)
        precision_score = ((y_test == y_pred) & (y_test == 1)).sum() / y_pred.sum() # this should be very high (>90)

        # the recall score is modified, where instead of the number of noisy samples in the current epoch, we use the maximal of all epochs.  
        recall_score = ((y_test == y_pred) & (y_test == 1)).sum() / total_num_noisy # this should be as high as possible.
        
        precisions.append(precision_score)
        recalls.append(recall_score)

    return np.asarray(precisions), np.asarray(recalls), np.asarray(thresholds)



def get_score_from_df(dataset, metric, epoch, noise_flag_name, total_num_noisy):   

    # set y_test, and y_pred_proba
    y_test = dataset[noise_flag_name].values # noisy are 1, clean are 0
    y_pred_proba_values = dataset[metric].values  # metric values will serve as prediction probabilities (scores) in this binary classification task 

    y_test = y_test[~np.isnan(y_pred_proba_values)]
    y_pred_proba = y_pred_proba_values[~np.isnan(y_pred_proba_values)]

    y_test = y_test[~np.isinf(y_pred_proba)]
    y_pred_proba = y_pred_proba[~np.isinf(y_pred_proba)]

    # normalize scores (min max normalization)
    minimum = min(y_pred_proba)
    maximum = max(y_pred_proba)
    if maximum == minimum:
        diff = 1
    else:
        diff = max(y_pred_proba) - minimum
    y_pred_proba_normalized =  (y_pred_proba - minimum) / diff # for later: maybe change this to 95% with clipping
    
    # calculate precisions and recalls at varying thresholds (both directions)
    precisions_right, recalls_right, thresholds_right = get_metrics_thresholds(y_test,  y_pred_proba, metric, 'right', epoch=epoch, total_num_noisy=total_num_noisy)
    average_precision_score_right = metrics.average_precision_score(y_test, y_pred_proba_normalized)
    # noisy are 1, clean are 0
    # direction: right

    precisions_left, recalls_left, thresholds_left = get_metrics_thresholds(y_test,  y_pred_proba, metric, 'left', epoch=epoch, total_num_noisy=total_num_noisy)  
    average_precision_score_left = metrics.average_precision_score(y_test, 1 - y_pred_proba_normalized)
    # noisy are 0, clean are 1
    # direction: left

    # check which direction has a higher overall detection potential (higher average precison - auc)
    if average_precision_score_left > average_precision_score_right:
        direction = 'left'
    else:
        direction = 'right'

    f05 = 1.25 * precisions_left * recalls_left / (0.25 * precisions_left + recalls_left)
    f05_left = np.nan_to_num(f05)

    f1 = 2 * precisions_left * recalls_left / (precisions_left + recalls_left)
    f1_left = np.nan_to_num(f1)

    f2 = 5 * precisions_left * recalls_left / (4 * precisions_left + recalls_left)
    f2_left = np.nan_to_num(f2)

    # thresholds = thresholds_left
    #else:
    f05 = 1.25 * precisions_right * recalls_right / (0.25 * precisions_right + recalls_right)
    f05_right = np.nan_to_num(f05)

    f1 = 2 * precisions_right * recalls_right / (precisions_right + recalls_right)
    f1_right = np.nan_to_num(f1)

    f2 = 5 * precisions_right * recalls_right / (4 * precisions_right + recalls_right)
    f2_right = np.nan_to_num(f2)
    # thresholds = thresholds_right

    return {'left': {'f05':f05_left, 'f1':f1_left,'f2':f2_left, 'thresholds':thresholds_left},
            'right': {'f05':f05_right, 'f1':f1_right,'f2':f2_right, 'thresholds':thresholds_right},
            'direction': direction
            }

def output_config(category, metric, f_type, epoch, threshold, direction, mode, config, corpus_name):

    # create config path
    config_path = config['paths']['configs_path'][mode] + os.sep + corpus_name       
    config_path += os.sep + 'category'+category['id'] + os.sep + metric + os.sep + f_type 

    if not os.path.exists(config_path):
        os.makedirs(config_path)

    # define the base config properties
    base_config = {

    "experiment_name": "relabel_cat"+category['id'],
    
    "paths": {
        "resources_path": f"{config['paths']['resources_path']}/relabel_cat{category['id']}/",
        "data_path":config['paths']['data_path'],
        "train_filename_extension" :config['paths']['train_filename_extension'],
        "dev_filename_extension" :config['paths']['dev_filename_extension'],
        "test_filename_extension" :config['paths']['test_filename_extension'],
        "baseline_paths":{
            "EE":config['paths']['baseline_paths']['EE'],
            "standard":config['paths']['baseline_paths']['standard'],
        }
    },
    "parameters": {
        "batch_size":config['parameters']['batch_size'],
        "learning_rate":config['parameters']['learning_rate'],
        "num_epochs":config['parameters']['num_epochs'],
        "model":config['parameters']['model'],
        "monitor_test":config['parameters']['monitor_test'],
        "scheduler":config['parameters']['scheduler'],
        "metrics_mode":config['parameters']['metrics_mode'],
        "model_reinit":config['parameters']['model_reinit'],
        "decoder_init":config['parameters']['decoder_init'],
        "modify_category1":False,
        "modify_category2":False,
        "modify_category3":False,
        "modify_category4":False,
    },
    "corpora" : config['corpora'],
    "seeds":config['seeds']
    }


    base_config['parameters']['seq_tagger_mode'] = mode

    # add current category modification parameters with 'mask' option
    base_config['parameters']['modify_category'+category['id']] = {
                                                    'epoch_change': str(epoch),
                                                     'metric':str(metric),
                                                     'f_type':f_type,
                                                     'threshold':str(threshold),
                                                     'direction':direction,
                                                     'modification':'mask'
                                                     }
    with open(config_path + os.sep + 'mask.config', 'w') as fp:
        json.dump(base_config, fp, indent=4)


    if int(category['id']) == 2 or int(category['id']) == 4:
        # add current category modification parameters with 'relabel' option
        # *only for categories 2 and 4 (because we have an alternative label there: the predicted one)
        base_config['parameters']['modify_category'+category['id']] = {
                                                        'epoch_change': str(epoch),
                                                        'metric':str(metric),
                                                        'f_type':f_type,
                                                        'threshold':str(threshold),
                                                        'direction':direction,
                                                        'modification':'relabel'
                                                        }
        with open(config_path + os.sep + 'relabel.config', 'w') as fp:
            json.dump(base_config, fp, indent=4)

def write_output(file, metric, f_types, score, epoch, threshold, direction, category, mode, config, corpus_name=''):
    f_type_str = '_'.join(f_types) if isinstance(f_types, list) else f_types
    file.write(f"{metric}, {f_type_str}, {score}, {epoch}, {threshold}, {direction}\n")
    # output_config(category, metric, f_type_str, epoch, threshold, direction, mode, config, corpus_name)


def optimize_F1s(config):
    ''' 

    This function finds the optimal parameter sets for detecting incorrect token labels, for each sample metric. 
    Across multiple corpora
    Each parameter set consists of:
        -   epoch
        -   threshold
        -   direction (< or >, i.e. left or right)
    The best parameter set is determined based on the highest F-scores for the binary classification task (classifying incorrect and correct tokens).
    This includes F05, F1 and F2 scores, which means that for each metric 3 sets of parameters are determined. 

    The optimal parameter sets are printed out to .csv files, where each category has a separate file.
    In addition, for each parameter set, a corresponding json config file is created. 

    These .json files can be used to run the relabelling/masking experiment from the main() function.

    '''

    corpora = config['source_corpora']
    
    # get general parameters from the config
    seq_tagger_modes = config['parameters']['modes']
    max_epochs = int(config['parameters']['num_epochs'])
    sample_metrics = config['sample_metrics']
    seeds = config['seeds']
    results_path = config['paths']['results_tables_path']

    # set flags and paths
    correct_prediction_flag_name = 'correct_prediction_flag'
    noise_flag_name = 'noisy_flag'
    resources_paths = {}
    resources_paths['EE'] = [f'{seed}_with_init-0.3/' for seed in seeds]
    resources_paths['standard'] = [f'{seed}/' for seed in seeds]

    for mode in seq_tagger_modes:

        # get the maximum number of noisy samples for each category

        for cat in CATEGORIES:
            cat['max_num_noisy'] = {corpus: {seed: 0 for seed in seeds} for corpus in corpora}

        for corpus in corpora:
            base_path = config['paths']['baseline_paths'][mode] + os.sep + corpus 

            filepath = base_path + resources_paths[mode][0]+'epoch_log'+'_0.log'

            # this is a bit useless and should be removed. start epoch should be the same for all corpora (and seeds)
            if not os.path.exists(filepath):
                start_epoch = 1
            else:
                start_epoch = 0
            
            for seed, resources_path in zip(seeds, resources_paths[mode]):

                path = base_path + os.sep+ resources_path 

                for i in [str(i) for i in range(start_epoch, max_epochs)]:

                    filepath = path+'epoch_log'+'_'+i+'.log'
                    epoch_log_df = pd.read_csv(filepath,  delimiter='\t', header=0, quoting=csv.QUOTE_NONE)

                    epoch_log_df[correct_prediction_flag_name] = epoch_log_df['predicted'] == epoch_log_df['noisy']

                    for category in CATEGORIES:

                        if category['observed_label'] == 'O':
                            category_epoch_log_df = epoch_log_df[(epoch_log_df[correct_prediction_flag_name] == category['correct_prediction_flag'])  & (epoch_log_df['noisy']=='O')]
                        else:
                            category_epoch_log_df = epoch_log_df[(epoch_log_df[correct_prediction_flag_name] == category['correct_prediction_flag'])  & (epoch_log_df['noisy']!='O')]

                        total_num_noisy = category_epoch_log_df['noisy_flag'].sum()

                        if total_num_noisy > category['max_num_noisy'][corpus][seed]:
                            category['max_num_noisy'][corpus][seed] = total_num_noisy
            
        # iterate over categories

        for category in CATEGORIES:
            all_threshold_scores = {}

            # set the scores and thresholds if epoch 0 is missing
            for score in F_SCORE_NAMES:
                all_threshold_scores[score] = {
                    metric: {
                        'scores' : [], 'thresholds':[]
                    } 
                    for metric in sample_metrics[mode]
                }

                for i in range(0, start_epoch):
                    for metric in sample_metrics[mode]:
                        all_threshold_scores[score][metric]['scores'].append(0)
                        all_threshold_scores[score][metric]['thresholds'].append(0)

            # iterate over epochs
            for i in [str(i) for i in range(start_epoch, max_epochs)]:
                
                # initialize directions and threshold_scores dictionaries
                directions = {metric: [] for metric in sample_metrics[mode]}

                threshold_scores = {}
                for score in F_SCORE_NAMES:
                    threshold_scores[score] = {direction:{metric: [] for metric in sample_metrics[mode]} for direction in ['left', 'right']}
                thresholds = {direction: {metric: [] for metric in sample_metrics[mode]} for direction in ['left', 'right']}

                # change in this new function: instead of only averaging over seeds and finding the maximum, we average over seeds AND CORPORA and find the maximum.
                # everything else stays the same;
                for corpus in corpora:

                    # iterate over seeds
                    for seed, resources_path in zip(seeds, resources_paths[mode]):
                        base_path = config['paths']['baseline_paths'][mode] + os.sep + corpus 

                        # read epoch log file
                        path = base_path + os.sep +resources_path 
                        filepath = path+'epoch_log'+'_'+i+'.log'
                        epoch_log_df = pd.read_csv(filepath,  delimiter='\t', header=0, quoting=csv.QUOTE_NONE)
                        epoch_log_df[correct_prediction_flag_name] = epoch_log_df['predicted'] == epoch_log_df['noisy']

                        # select the data subset that is in the corresponding epoch
                        if category['observed_label'] == 'O':
                            category_epoch_log_df = epoch_log_df[(epoch_log_df[correct_prediction_flag_name] == category['correct_prediction_flag'])  & (epoch_log_df['noisy']=='O')]
                        else:
                            category_epoch_log_df = epoch_log_df[(epoch_log_df[correct_prediction_flag_name] == category['correct_prediction_flag'])  & (epoch_log_df['noisy']!='O')]

                        # get total number of epochs (this can be different from current epoch, only for EE mode and if metrics were calculated during decoder init)
                        total_epoch = category_epoch_log_df['mild'].max()

                        # calculate f_scores for each sample metric

                        for metric in sample_metrics[mode]:
                            result = get_score_from_df(category_epoch_log_df, metric, epoch=total_epoch, noise_flag_name=noise_flag_name, total_num_noisy=category['max_num_noisy'][corpus][seed]) #list of 10 (over thresholds)

                            for f_type in F_SCORE_NAMES:
                                threshold_scores[f_type]['left'][metric].append(result['left'][f_type]) # this is a list of lists 
                                threshold_scores[f_type]['right'][metric].append(result['right'][f_type]) # this is a list of lists 


                            thresholds['left'][metric].append(result['left']['thresholds']) # this is a list of lists
                            thresholds['right'][metric].append(result['right']['thresholds']) # this is a list of lists

                            directions[metric].append(result['direction'])




                # iterate over metrics
                for metric in sample_metrics[mode]:
                    
                    counter = Counter(directions[metric])

                    direction, count = counter.most_common()[0]

                    # calculate minimum number of thresholds 
                    # (in case the number of thresholds differs among seeds)
                    min_len = len(thresholds[direction][metric][0])
                    for l in thresholds[direction][metric][1:]:
                        if len(l) < min_len:
                            min_len = len(l)

                    # take only up to min_len thresholds (from the seeds that have more)
                    thresholds_list = np.asarray([l[:min_len] for l in thresholds[direction][metric]]).mean(axis=0)

                    # iterate over the three f_score types
                    for f_score_type in F_SCORE_NAMES:

                        # take only up to min_len scores
                        scores = np.array([l[:min_len] for l in threshold_scores[f_score_type][direction][metric]])

                        # average the f_scores over the seeds
                        scores = scores.mean(axis=0)

                        if len(scores) == 0:
                            # if the list of scores is empty, set them and the other parameters to 0
                            all_threshold_scores[f_score_type][metric]['scores'].append(0)
                            all_threshold_scores[f_score_type][metric]['thresholds'].append(0)
                            all_threshold_scores[f_score_type][metric]['direction'] = direction
                        else:
                            all_threshold_scores[f_score_type][metric]['scores'].append(max(scores))
                            all_threshold_scores[f_score_type][metric]['thresholds'].append(thresholds_list[np.argmax(scores)])
                            all_threshold_scores[f_score_type][metric]['direction'] = direction

            
            # open the .csv for current category 
            merged_corpora_names = '_'.join(corpora)
            filepath = results_path + os.sep + merged_corpora_names + os.sep + mode+'_mode'

            if not os.path.exists(filepath):
                os.makedirs(filepath)

            optimize_F1s_output_file = open(filepath + os.sep+'optimal_F1s_category'+category['id']+'.csv','w')
            optimize_F1s_output_file.write('metric, f_score, score, epoch, threshold, direction\n')

            optimize_F1s_output_file_parameters_merged = open(filepath + os.sep+'optimal_F1s_category'+category['id']+'_parameters_merged.csv','w')
            optimize_F1s_output_file_parameters_merged.write('metric, f_score, score, epoch, threshold, direction\n')

            # iterate over metrics
            for metric in sample_metrics[mode]:
                epochs = []
                thresholds = []
                directions = []
                scores = []
                for f_type in F_SCORE_NAMES:

                    score = np.max(all_threshold_scores[f_type][metric]['scores'])
                    threshold = all_threshold_scores[f_type][metric]['thresholds'][np.argmax(all_threshold_scores[f_type][metric]['scores'])]
                    epoch = np.argmax(all_threshold_scores[f_type][metric]['scores'])
                    direction = all_threshold_scores[f_type][metric]['direction']
                    epochs.append(epoch)
                    thresholds.append(threshold)
                    directions.append(direction)
                    scores.append(score)
                    '''
                    uncomment the following code to print a full table (where duplicate parameter sets are NOT merged), which includes all actual f score values
                    '''
                    write_output(optimize_F1s_output_file, metric, f_type, score, epoch, threshold, direction, category, mode, config, corpus_name)
                    # optimize_F1s_output_file.write(f'{metric}, {f_type}, {score}, {epoch}, {threshold}, {direction}\n')
                    # output_config(category, metric,  f_type, epoch, threshold, direction, mode)


                ''' 
                the following code prints a reduced table (where duplicate parameter sets are merged)
                '''

                # indices:
                # 0 - f05
                # 1 - f1
                # 2 - f2      
                
                # in the case of an overlapping parameter set, only the first f score is printed out (according to the indices above) 
                indices = {0,1,2}
                pairs = [(0, 1), (1, 2), (0, 2)]
                for i, j in pairs:
                    if epochs[i] == epochs[j] and thresholds[i] == thresholds[j] and directions[i] == directions[j]:

                        remaining = indices - {i, j}
                        k = remaining.pop()
                        
                        if epochs[i] == epochs[k] and thresholds[i] == thresholds[k] and directions[i] == directions[k]:
                            # all three parameter sets are the same 
                            # (only f05 score is printed out)
                            write_output(optimize_F1s_output_file_parameters_merged, metric, F_SCORE_NAMES, scores[0], epochs[0], thresholds[0], directions[0], category, mode, config, corpus_name)
                        else:
                            # two parameter sets are the same, one is different
                            write_output(optimize_F1s_output_file_parameters_merged, metric, [F_SCORE_NAMES[i], F_SCORE_NAMES[j]], scores[i], epochs[i], thresholds[i], directions[i], category, mode, config, corpus_name)
                            write_output(optimize_F1s_output_file_parameters_merged, metric, F_SCORE_NAMES[k], scores[k], epochs[k], thresholds[k], directions[k], category, mode, config, corpus_name)
                        break
                else:
                    # all three parameter sets are different
                    for i in list(indices):
                        write_output(optimize_F1s_output_file_parameters_merged, metric, F_SCORE_NAMES[i], scores[i], epochs[i], thresholds[i], directions[i], category, mode, config, corpus_name)

def calculate_correlations(config):
    ''' 
    This function calculates the correlation between the detection F-scores and the test scores for each category.
    The correlations are calculated for each mode separately. 
    The results are printed out to .csv files. 
    '''

    seq_tagger_modes = config['parameters']['modes']
    max_epochs = int(config['parameters']['num_epochs'])
    sample_metrics = config['sample_metrics']
    base_paths = {key: config['paths']['baseline_paths'][key] for key in seq_tagger_modes}
    seeds = config['seeds']
    results_path = config['paths']['results_tables_path']
    corpora = config['corpora']
    source_corpus = config['source_corpus']

    for corpus in corpora:
        # open the .csv for current corpus 

        filepath = f'{results_path}/source_{source_corpus}_target_{corpus}/correlations'
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        correlations_output_file = open(filepath + os.sep+'correlations.csv','w')
        correlations_output_file.write('category, f_type, modification, correlation\n')

        for category in CATEGORIES:
                if 'category'+category['id'] not in config['categories']:
                    continue

                # read the optimal F1s file
                optimal_F1s_file = open(f"{results_path}/source_{source_corpus}_target_{corpus}/final_tables/category{category['id']}_final_table.csv",'r')
                lines = optimal_F1s_file.readlines()
                optimal_F1s_file.close()

                for modif in ['mask', 'relabel']:
                    for f_type in ['f05','f1','f2']:
                        # get the f_scores and the corresponding metrics
                        f_scores = []
                        test_scores = []
                        for line in lines[1:]:
                            metric, f_score, modification, score, epoch, threshold, direction, noise_share, test_score, std_test_score  = line.split(',')
                            if f_type in f_score.strip() and modification.strip() == modif:
                                f_scores.append(float(score))
                                test_scores.append(float(test_score))
                        if len(f_scores) == 0:
                            continue
                        # calculate the correlation
                        correlation = np.corrcoef(f_scores, test_scores)[0,1]

                        # write the correlation to the file
                        correlations_output_file.write(f"category{category['id']}, {f_type}, {modif}, {correlation}\n")