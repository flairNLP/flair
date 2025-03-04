
import argparse
import json
from pipeline_sample_metrics_token_categories import *
from optimize_metric_parameters import *
from summarize_scores_sample_metrics import *
from merge_optimal_F_score_tables import *
import os

def run(config):

    seeds = [int(seed) for seed in config['seeds']]
    corpora = config['corpora']

    for corpus_name in corpora:

        for mode in config['parameters']['modes']:
            temp_f1_scores = []
            for seed in seeds:
                baseline_path, score = run_baseline(mode, seed,  corpus_name, config, int(config['parameters']['num_epochs']))

                temp_f1_scores.append(score)

            with open(baseline_path + os.sep + "test_results.tsv", "w", encoding='utf-8') as f:
                f.write("params\tmean\tstd\n")
                label = f"{str(config['parameters']['batch_size'])}_{str(config['parameters']['learning_rate'])}"
                f.write(f"{label} \t{np.mean(temp_f1_scores)!s} \t {np.std(temp_f1_scores)!s} \n")

        optimize_F1s(config, corpus_name=corpus_name)

        for mode in config['parameters']['modes']:
            config_path = config['paths']['configs_path'][mode]

            for dirpath,_,filenames in os.walk(config_path):
                for f in filenames:
                    config_filepath = os.path.abspath(os.path.join(dirpath, f))

                    with open(config_filepath) as json_file:
                        config = json.load(json_file)

                    main(config)
        summarize_test_scores(config['paths']['results_tables_path'], corpus_name)

        merge_tables(f"{config['paths']['results_tables_path']}/{corpus_name}", config['parameters']['modes']) 


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-c", "--config", help="filename with experiment configuration")

    args = argParser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)

    run(config)

