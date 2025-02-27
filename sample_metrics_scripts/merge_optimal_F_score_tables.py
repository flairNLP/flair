import pandas as pd
import os

base_path2 = '.'

for category in ['1','2','3','4']:#, '2','3']:#,'3','4']:
    data2 = pd.read_csv(base_path2+os.sep+'optimal_category'+category+'_test_scores.csv',header = 0, index_col=[0,1, 2])

    full_data = None
    for mode in ['standard', 'EE']:
        base_path1 = f'./noisebench_baselin/category0/{mode}_/noise_crowd'

        data1 = pd.read_csv(base_path1+os.sep+'optimal_testF1s_merged_category'+category+'.csv',header = 0, index_col=[0,1])
        print(data1)

        if full_data is None:
            full_data = pd.merge(data1, data2, left_index=True, right_index=True)
        else:
            full_data = pd.concat([full_data, pd.merge(data1, data2, left_index=True, right_index=True)], axis = 0)
    full_data.to_csv(f'category{category}_merged_optimal_table.csv',index=True,header=True, float_format='%.3f')
