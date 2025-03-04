
import os
import csv
import numpy as np
import pandas as pd
from sklearn import metrics
import json
import seaborn as sns
import matplotlib.pyplot as plt

# separate file for each category
# rows: metrics 
# columns: F score type, type, threshold, epoch and </> (direction), seq_tagger_mode.


categories = [
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
    print(metric)
    print(y_pred_proba)
    print(len(y_pred_proba))
    print(total_num_noisy)

    if metric in ['msp','BvSB','confidence', 'correctness','iter_norm']:
        thresholds = np.arange(0, 1, 0.1)
    elif metric in ['variability']:
        thresholds = np.arange(0, 0.5, 0.05)
    elif metric in [ 'tac', 'pd', 'fl', 'tal' ]:
        thresholds = np.arange(0, 25, 1)
    elif metric in [ 'cross_entropy', 'entropy', 'le', 'pehist' ]:
        minimum = 0
        maximum = max(y_pred_proba)
        if metric != 'pehist':
            thresholds = np.arange(minimum, maximum, 0.1)
        else:
            thresholds = np.arange(minimum, maximum, 0.05)
    elif metric == 'mild':
        # todo: fix.
        thresholds = np.arange(-epoch, epoch, 1)
        # print(thresholds)
        # input()
        # print(y_pred_proba)
        # input()
    elif metric == 'mild_f':
        # todo: fix.
        thresholds = np.arange(1, epoch +1 , 1)
    else:
        # todo: fix.
        thresholds = np.arange(1, epoch+1, 1)

    prec, rec = [], []
    print(y_test)
    # fig, ax = plt.subplots()
    # sns.histplot(pd.concat([pd.Series(y_pred_proba, name='val'), pd.Series(y_test, name='flag')], axis=1), binwidth=0.1,binrange=(0,1), hue='flag',x='val', ax=ax)
    # ax.set_ylim((0,1000))
    # plt.show()
    for th in thresholds:
        if direction == 'left':
            y_pred = np.where(y_pred_proba < th, 1, 0)
        else:
            y_pred = np.where(y_pred_proba > th, 1, 0)
        prec_score = ((y_test == y_pred) & (y_test == 1)).sum() / y_pred.sum() # this should be very high (>90)
        rec_score = ((y_test == y_pred) & (y_test == 1)).sum() / total_num_noisy #this should be as high as possible.
        f05 = 1.25 * prec_score * rec_score / (0.25 * prec_score + rec_score)
        
        prec.append(prec_score)
        rec.append(rec_score)

    return np.asarray(prec), np.asarray(rec), np.asarray(thresholds)



def get_score_from_df(dataset, metric, epoch, noise_flag_name, total_num_noisy):           
    y_test = dataset[noise_flag_name].values # noisy are 1, clean are 0
    y_pred_proba_values = dataset[metric].values
    #print(y_test)
    noisy = sum(y_test)
    clean = len(y_test) - noisy

    y_test = y_test[~np.isnan(y_pred_proba_values)]

    y_pred_proba = y_pred_proba_values[~np.isnan(y_pred_proba_values)]

    y_test = y_test[~np.isinf(y_pred_proba)]
    y_pred_proba = y_pred_proba[~np.isinf(y_pred_proba)]

    minimum = min(y_pred_proba)
    maximum = max(y_pred_proba)

    if maximum == minimum:
        diff = 1
    else:
        diff = max(y_pred_proba) - minimum

    y_pred_proba_normalized =  (y_pred_proba - minimum) / diff
    # TODO: change this to 95% with clipping...

    #print(y_pred_proba)
    prec1, rec1, thresholds1 = get_metrics_thresholds(y_test,  y_pred_proba, metric, 'right', epoch=epoch, total_num_noisy=total_num_noisy)
    print(prec1)
    print(rec1)
    auc1 = metrics.average_precision_score(y_test, y_pred_proba_normalized)
    # noisy are 1, clean are 0
    # direction: right

    prec2, rec2, thresholds2 = get_metrics_thresholds(y_test,  y_pred_proba, metric, 'left', epoch=epoch, total_num_noisy=total_num_noisy)
    print(prec2)
    print(rec2)
    print(len(prec1))
    print(len(rec1))
    y_pred_proba2_normalized =  1-y_pred_proba_normalized
    auc2 = metrics.average_precision_score(y_test, y_pred_proba2_normalized)
    # noisy are 0, clean are 1
    # direction: left

    if auc2 > auc1:
        f05 = 1.25 * prec2 * rec2 / (0.25 * prec2+rec2)
        f05 = np.nan_to_num(f05)

        f1 = 2 * prec2 * rec2 / (prec2+rec2)
        f1 = np.nan_to_num(f1)

        f2 = 5 * prec2 * rec2 / (4 * prec2+rec2)
        f2 = np.nan_to_num(f2)

        direction = 'left'
        thresholds = thresholds2
    else:
        f05 = 1.25 * prec1 * rec1 / (0.25 * prec1+rec1)
        f05 = np.nan_to_num(f05)

        f1 = 2 * prec1 * rec1 / (prec1+rec1)
        f1 = np.nan_to_num(f1)

        f2 = 5 * prec1 * rec1 / (4 * prec1+rec1)
        f2 = np.nan_to_num(f2)
        thresholds = thresholds1
        direction = 'right'

    return {'f05':f05, 'f1':f1,'f2':f2, 'thresholds':thresholds, 'direction':direction}

def output_config(category, metric, f_type, score, epoch, threshold, direction, mode, config):

    config_path = config['paths']['configs_path'][mode]
    
    if not os.path.exists(config_path ):
        os.makedirs(config_path )

    config_path += os.sep + 'category'+category['id'] + os.sep + metric + os.sep + f_type 

    if not os.path.exists(config_path):
        os.makedirs(config_path)

    base_config = {

    "experiment_name": "relabel_cat"+category['id'],
    
    "paths": {
        "resources_path": f"{config['paths']['results_tables_path']}/resources/relabel_cat{category['id']}/",
        "data_path":"../../NoiseBench/data/noisebench/.nessieformat/",
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
        "modify_category1":False,
        "modify_category2":False,
        "modify_category3":False,
        "modify_category4":False,
    },
    "corpora" : [
        "noise_crowd"
    ],
    "seeds":config['seeds']
    }

    base_config['parameters']['seq_tagger_mode'] = mode

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

def optimize_F1s(config, corpus_name):
    '''
    Categories:

    1. correct prediction and predicted label is O - (0,0)
    2. correct prediction and predicted label is not O - (1,0)
    3. incorrect prediction and predicted label is O - (0,1)
    4. incorrect prediction and predicted label is not O - (1,1) 
    '''

    seq_tagger_modes = config['parameters']['modes']
    metrics_mode = config['parameters']['metrics_mode']
    max_epochs = config['parameters']['num_epochs']

    sample_metrics = config['sample_metrics']

    correct_prediction_flag_name = 'correct_prediction_flag'
    noise_flag_name = 'noisy_flag'

    base_paths = {key: config['paths']['baseline_paths'][key] + corpus_name for key in seq_tagger_modes}

    seeds = config['seeds']

    exp_paths = {}
    exp_paths['EE'] = [f'{seed}_with_init-0.3/' for seed in seeds]#, '42_with_init-0.3/','100_with_init-0.3/']  1_with_init-0.3
    exp_paths['standard'] = [f'{seed}/' for seed in seeds]

    results_path = config['paths']['results_tables_path']

    for mode in seq_tagger_modes:

        filepath = base_paths[mode] + exp_paths[mode][0]+'epoch_log'+'_0.log'

        if not os.path.exists(filepath):
            start_index = 1
        else:
            start_index = 0

        for cat in categories:
            cat['max_num_noisy'] = {seed: 0 for seed in seeds}

        for seed, exp_path in zip(seeds, exp_paths[mode]):
            path = base_paths[mode] + os.sep+ exp_path 

            for i in [str(i) for i in range(start_index, max_epochs)]:

                filepath = path+'epoch_log'+'_'+i+'.log'
                df = pd.read_csv(filepath,  delimiter='\t', header=0, quoting=csv.QUOTE_NONE)

                df[correct_prediction_flag_name] = df['predicted'] == df['noisy']
                # print('full df len ')
                # print(len(df))
                # input()
                for category in categories:

                    if category['observed_label'] == 'O':
                        dataset = df[(df[correct_prediction_flag_name] == category['correct_prediction_flag'])  & (df['noisy']=='O')]
                    else:
                        dataset = df[(df[correct_prediction_flag_name] == category['correct_prediction_flag'])  & (df['noisy']!='O')]

                    total_num_noisy = dataset['noisy_flag'].sum()

                    if total_num_noisy > category['max_num_noisy'][seed]:
                        category['max_num_noisy'][seed] = total_num_noisy

        for category in categories:
            print(category)            
            f_scores = ['f05','f1','f2']
            all_threshold_scores = {}

            for score in f_scores:
                all_threshold_scores[score] = {
                    metric: {
                        'scores' : [], 'thresholds':[]
                    } 
                    for metric in sample_metrics[mode]
                }

                for i in range(0, start_index):
                    for metric in sample_metrics[mode]:
                        all_threshold_scores[score][metric]['scores'].append(0)
                        all_threshold_scores[score][metric]['thresholds'].append(0)


            for i in [str(i) for i in range(start_index, max_epochs)]:
                print('epoch')
                print(i)
                directions = {metric: [] for metric in sample_metrics[mode]}

                threshold_scores = {}
                for score in f_scores:
                    threshold_scores[score] = {metric: [] for metric in sample_metrics[mode]}

                thresholds = {metric: [] for metric in sample_metrics[mode]}

                for seed, exp_path in zip(seeds, exp_paths[mode]):
                    print('seed')
                    print(seed)
                    print(category)
                    path = base_paths[mode] + os.sep +exp_path 

                    filepath = path+'epoch_log'+'_'+i+'.log'
                    df = pd.read_csv(filepath,  delimiter='\t', header=0, quoting=csv.QUOTE_NONE)
                    df[correct_prediction_flag_name] = df['predicted'] == df['noisy']

                    if category['observed_label'] == 'O':
                        dataset = df[(df[correct_prediction_flag_name] == category['correct_prediction_flag'])  & (df['noisy']=='O')]
                    else:
                        dataset = df[(df[correct_prediction_flag_name] == category['correct_prediction_flag'])  & (df['noisy']!='O')]

                    total_epoch = dataset['mild'].max()

                    for metric in sample_metrics[mode]:
                        result = get_score_from_df(dataset, metric, epoch=total_epoch, noise_flag_name=noise_flag_name, total_num_noisy=category['max_num_noisy'][seed]) #list of 10 (over thresholds)

                        for f_type in f_scores:
                            threshold_scores[f_type][metric].append(result[f_type])

                        thresholds[metric].append(result['thresholds'])
                        directions[metric].append(result['direction'])


                # avg over seeds
                for metric in sample_metrics[mode]:
                    print(metric)
                    print(thresholds[metric])
                    min_len = len(thresholds[metric][0])
                    for l in thresholds[metric][1:]:
                        if len(l) < min_len:
                            min_len = len(l)
                    thresholds_list = np.asarray([l[:min_len] for l in thresholds[metric]]).mean(axis=0)
                    print(directions[metric])
                    # punish if directions don't match

                    for f_score in f_scores:
                        scores = np.array([l[:min_len] for l in threshold_scores[f_score][metric]])
                        scores = scores.mean(axis=0)
                        if len(scores) == 0:
                            # get maxima over thresholds
                            all_threshold_scores[f_score][metric]['scores'].append(0)
                            all_threshold_scores[f_score][metric]['thresholds'].append(0)
                            all_threshold_scores[f_score][metric]['direction'] = directions[metric][0]
                        else:
                            # get maxima over thresholds
                            all_threshold_scores[f_score][metric]['scores'].append(max(scores))
                            all_threshold_scores[f_score][metric]['thresholds'].append(thresholds_list[np.argmax(scores)])
                            
                            all_threshold_scores[f_score][metric]['direction'] = directions[metric][0]
                            # change this: not dicts, but arrays: all_threshold_scores_f05[metric]['scores'], all_threshold_scores_f05[metric]['thresholds'], all_threshold_scores_f05[metric]['direction'] (this can also be one variable)

            # final_scores_and_thresholds.

            # max_over_epochs #per_metric 
            # and
            # output_results_file(category)
            
            filepath = results_path + os.sep + corpus_name + os.sep + mode+'_mode'

            if not os.path.exists(filepath):
                os.makedirs(filepath)

            optimize_F1s_output = open(filepath + os.sep+'optimal_F1s_category'+category['id']+'.csv','w')
            
            optimize_F1s_output.write('metric, f_score, score, epoch, threshold, direction\n')

            for metric in sample_metrics[mode]:
                epochs = []
                thresholds = []
                directions = []
                scores = []
                for f_type in f_scores:
                    print(all_threshold_scores[f_type][metric]['scores'])
                    print(all_threshold_scores[f_type][metric]['thresholds'])
                    print(all_threshold_scores[f_type][metric]['direction'])

                    score = np.max(all_threshold_scores[f_type][metric]['scores'])
                    threshold = all_threshold_scores[f_type][metric]['thresholds'][np.argmax(all_threshold_scores[f_type][metric]['scores'])]
                    epoch = np.argmax(all_threshold_scores[f_type][metric]['scores'])
                    direction = all_threshold_scores[f_type][metric]['direction']
                    epochs.append(epoch)
                    thresholds.append(threshold)
                    directions.append(direction)
                    scores.append(score)
                    # uncomment to get full table with actual f score values
                    # optimize_F1s_output.write(f'{metric}, {f_type}, {score}, {epoch}, {threshold}, {direction}\n')
                    # output_config(category, metric,  f_type, score, epoch, threshold, direction, mode)

                # uncomment to get reduced table with merged duplicate parameter sets
                # todo: fix
                if epochs[0] == epochs[1] and thresholds[0] == thresholds[1] and directions[0] == directions[1]:
                    if epochs[2] == epochs[1] and thresholds[2] == thresholds[1] and directions[2] == directions[1]: #123
                        optimize_F1s_output.write(f"{metric}, {'_'.join([f_type for f_type in f_scores])}, {scores[0]}, {epochs[0]}, {thresholds[0]}, {directions[0]}\n")
                        output_config(category, metric,  '_'.join([f_type for f_type in f_scores]), scores[0], epochs[0], thresholds[0], directions[0], mode, config)
                    else: #12, 3
                        optimize_F1s_output.write(f"{metric}, {'_'.join([f_type for f_type in f_scores[0:2]])}, {scores[0]}, {epochs[0]}, {thresholds[0]}, {directions[0]}\n")
                        output_config(category, metric,  '_'.join([f_type for f_type in f_scores[0:2]]), scores[0], epochs[0], thresholds[0], directions[0], mode, config)

                        optimize_F1s_output.write(f"{metric}, {f_scores[2]}, {scores[2]}, {epochs[2]}, {thresholds[2]}, {directions[2]}\n")
                        output_config(category, metric,  f_scores[2], scores[2], epochs[2], thresholds[2], directions[2], mode, config)

                elif epochs[2] == epochs[1] and thresholds[2] == thresholds[1] and directions[2] == directions[1]: #1, 23
                    optimize_F1s_output.write(f"{metric}, {'_'.join([f_type for f_type in f_scores[1:3]])},  {scores[1]}, {epochs[2]}, {thresholds[2]}, {directions[2]}\n")
                    output_config(category, metric,  '_'.join([f_type for f_type in f_scores[1:3]]), score, epoch, threshold, direction, mode, config)

                    optimize_F1s_output.write(f"{metric}, {f_scores[0]}, {scores[0]}, {epochs[0]}, {thresholds[0]}, {directions[0]}\n")
                    output_config(category, metric,  f_scores[0], scores[0], epochs[0], thresholds[0], directions[0], mode, config)

                elif epochs[2] == epochs[0] and thresholds[2] == thresholds[0] and directions[2] == directions[0]: #2, 13
                    optimize_F1s_output.write(f"{metric}, {f_scores[0]+'_'+f_scores[2]},  {scores[0]}, {epochs[2]}, {thresholds[2]}, {directions[2]}\n")
                    output_config(category, metric,  f_scores[0]+'_'+f_scores[2], score, epoch, threshold, direction, mode, config)

                    optimize_F1s_output.write(f'{metric}, {f_scores[1]}, {scores[1]}, {epochs[1]}, {thresholds[1]}, {directions[1]}\n')
                    output_config(category, metric,  f_scores[1], scores[1], epochs[1], thresholds[1], directions[1], mode, config)   
                else:
                    optimize_F1s_output.write(f"{metric}, {f_scores[0]}, {scores[0]}, {epochs[0]}, {thresholds[0]}, {directions[0]}\n")
                    output_config(category, metric,  f_scores[0], scores[0], epochs[0], thresholds[0], directions[0], mode, config)

                    optimize_F1s_output.write(f"{metric}, {f_scores[1]}, {scores[1]}, {epochs[1]}, {thresholds[1]}, {directions[1]}\n")
                    output_config(category, metric,  f_scores[1], scores[1], epochs[1], thresholds[1], directions[1], mode, config)

                    optimize_F1s_output.write(f"{metric}, {f_scores[2]}, {scores[2]}, {epochs[2]}, {thresholds[2]}, {directions[2]}\n")
                    output_config(category, metric,  f_scores[2], scores[2], epochs[2], thresholds[2], directions[2], mode, config)
                #input()

#optimize_F1s()