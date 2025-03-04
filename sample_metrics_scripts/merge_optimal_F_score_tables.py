import pandas as pd
import os

categories_ids = ['1','2','3','4']

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
