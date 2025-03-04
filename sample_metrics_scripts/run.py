
import argparse
import json
from pipeline_sample_metrics_token_categories import *
from optimize_metric_parameters import *
from summarize_scores_sample_metrics import *
from merge_optimal_F_score_tables import *
import os

def run(config):

    seeds = [int(seed) for seed in config['parameters']['seeds']]
    corpora = config['parameters']['corpora']

    for corpus_name in corpora:

        for mode in config['parameters']['modes']:
            for seed in seeds:
                baseline_path = run_baseline(mode, seed,  corpus_name, config, int(config['parameters']['num_epochs']))

        optimize_F1s(config, corpus_name=corpus_name)

        for mode in config['parameters']['modes']:
            for config_path in config['parameters']['configs_path'][mode]:
                for dirpath,_,filenames in os.walk(config_path):
                    for f in filenames:
                        config_filepath = os.path.abspath(os.path.join(dirpath, f))

                        with open(config_filepath) as json_file:
                            config = json.load(json_file)
                            
                        main(config)

        summarize_test_scores(config['paths']['results_path'], corpus_name)

        merge_tables(f"{config['paths']['results_path']}/{corpus_name}") 


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-c", "--config", help="filename with experiment configuration")

    args = argParser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)

    run(config)

