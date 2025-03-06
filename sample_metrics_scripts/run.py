
import argparse
import json
from pipeline_sample_metrics_token_categories import *
from optimize_metric_parameters import *
from summarize_scores_sample_metrics import *
import os

def run(config, gpu=0):

    flair.device = torch.device("cuda:" + str(gpu))

    seeds = [int(seed) for seed in config['seeds']]
    corpora = config['corpora']

    for corpus_name in corpora:

        # 1. Run baseline for each mode: standard fine-tuning and early-exit fine-tuning
        for mode in config['parameters']['modes']:
            temp_f1_scores = []
            for seed in seeds:
                baseline_path, score = run_baseline(mode, seed,  corpus_name, config, int(config['parameters']['num_epochs']))

                temp_f1_scores.append(score)

            with open(baseline_path + os.sep + "test_results.tsv", "w", encoding='utf-8') as f:
                f.write("params\tmean\tstd\n")
                label = "f1"
                f.write(f"{label} \t{np.mean(temp_f1_scores)!s} \t {np.std(temp_f1_scores)!s} \n")

        # 2. Find optimal parameter sets for each sample metric, mode and category
        optimize_F1s(config, corpus_name=corpus_name)

        # 3. Run experiment (relabel or mask each category) based on the optimal parameter sets from 2. 
        for mode in config['parameters']['modes']:
            config_path = config['paths']['configs_path'][mode]

            for dirpath,_,filenames in os.walk(config_path):
                for f in filenames:
                    config_filepath = os.path.abspath(os.path.join(dirpath, f))

                    with open(config_filepath) as json_file:
                        experiment_config = json.load(json_file)

                    main(experiment_config)

        # 4. Summarize the test scores from 3. 
        summarize_test_scores(config['paths']['results_tables_path'], corpus_name)

        # 5. Merge the optimal parameter sets from 2. and the summarized test scores from 4.
        merge_tables(f"{config['paths']['results_tables_path']}/{corpus_name}", config['parameters']['modes']) 


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-c", "--config", help="filename with experiment configuration")
    argParser.add_argument("-g", "--gpu", help="set gpu id", default=0)

    args = argParser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)

    run(config, args.gpu)

