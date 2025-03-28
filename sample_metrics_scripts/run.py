import argparse
import json
from pipeline_sample_metrics_token_categories import *
from optimize_metric_parameters import *
from summarize_scores_sample_metrics import *
import os
import logging
import datetime
from pathlib import Path
import socket

def setup_logging(config):
    servername = socket.gethostname()
    device = torch.cuda.current_device()

    base_path = config['paths']['resources_path']+os.sep+'logs'
    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")+'_'+config['config_filepath']+'.log'

    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    file = base_path / filename
    file.touch(exist_ok=True)

    logger_flair = logging.getLogger("flair")
    logger_flair.handlers[0].setLevel(logging.WARNING)

    logger_experiment = logging.getLogger(__name__)
    logger_experiment.setLevel(level="INFO")

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

    fileHandler =    logging.FileHandler(file, mode="a", encoding="utf-8")

    fileHandler.setFormatter(logFormatter)
    logger_experiment.addHandler(fileHandler)


    logger_experiment.info(f"Hostname: {servername}")
    logger_experiment.info(f"GPU: {device}")
    logger_experiment.info(json.dumps(config, indent = 4))

    return logger_experiment


def run(config, gpu=0):

    logger_experiment = setup_logging(config)
    flair.device = torch.device("cuda:" + str(gpu))
    seeds = [int(seed) for seed in config['seeds']]
    corpora = config['corpora']

    for corpus_name in corpora:

        # 1. Run baseline for each mode: standard fine-tuning and early-exit fine-tuning
        for mode in config['parameters']['modes']:
            temp_f1_scores = []
            if not os.path.exists(f"{config['paths']['baseline_paths'][mode]}/{corpus_name}/{seeds[0]}") and not os.path.exists(f"{config['paths']['baseline_paths'][mode]}/{corpus_name}/{seeds[0]}_with_init-{config['parameters']['decoder_init']['lr']}"):
                logger_experiment.info(f"Running baseline for {corpus_name}, {mode} mode")
                for seed in seeds:
                    logger_experiment.info(f"Running seed {seed}")
                    # baseline paths don't have to be set beforehand
                    baseline_path, score = run_baseline(mode, seed, corpus_name, config,
                                                        int(config['parameters']['num_epochs']))

                    temp_f1_scores.append(score)

                with open(baseline_path +os.sep+corpus_name+ os.sep + "test_results.tsv", "w", encoding='utf-8') as f:
                    f.write("params\tmean\tstd\n")
                    label = "f1"
                    f.write(f"{label} \t{np.mean(temp_f1_scores)!s} \t {np.std(temp_f1_scores)!s} \n")
                logger_experiment.info(f"Baseline for {corpus_name}, {mode} mode for seeds {seeds} finished")

        # Optional: plot histograms of metrics for baseline runs, for each seed
        if config['plot_histograms']:
            logger_experiment.info(f"Plotting histograms and lineplot for {corpus_name} corpus")
            for mode in config['parameters']['modes']:
                plot_metric_distributions(base_path=f"{config['paths']['baseline_paths'][mode]}/{corpus_name}/",
                                          mode=mode, seeds=seeds, sample_metrics=config['sample_metrics'][mode],
                                          dset='train', max_epochs=int(config['parameters']['num_epochs']) + 1)
            plot_category_membership_through_epochs(base_paths = config['paths']['baseline_paths'] , corpus_name = corpus_name, seeds= seeds,dset = 'train', max_epochs=11)

        # 2. Find optimal parameter sets for each sample metric, mode and category
        logger_experiment.info(f"Optimizing F scores for {corpus_name}; saving to csv files and generating config json files")
        optimize_F1s(config, corpus_name=corpus_name)

    # 3. Run experiment (relabel or mask each category) based on the optimal parameter sets from 2. 
    for mode in config['parameters']['modes']:
        config_path = config['paths']['configs_path'][mode]
        logger_experiment.info(f"Running experiment for {mode} mode. Read configs from {config_path}.")

        for dirpath, _, filenames in os.walk(config_path):
            if any(s in dirpath for s in config['categories']):
                for f in filenames:
                    config_filepath = os.path.relpath(os.path.join(dirpath, f))

                    with open(config_filepath) as json_file:
                        experiment_config = json.load(json_file)
                    logger_experiment.info(f"Running category modification experiment... \n\t\tFrom config: {config_filepath}\n\t\tResources path: {experiment_config['paths']['resources_path']}\n\t\tData path:  {experiment_config['paths']['data_path']}\n\t\tFor following corpora: {experiment_config['corpora']}\n")

                    # here the experiment is ran for all noise types listed in the config file
                    main(experiment_config, gpu)
                    logger_experiment.info(f"Finished experiment from {config_filepath}")


    categories_ids = [cat[-1] for cat in config['categories']]
    for corpus_name in corpora:
        logger_experiment.info(f"Summarizing test scores for {corpus_name}")

        # 4. Summarize the test scores from 3. 
        summarize_test_scores(config['paths']['results_tables_path'], corpus_name, resources_path=config['paths']['resources_path'], categories_ids = categories_ids)

        # 5. Merge the optimal parameter sets from 2. and the summarized test scores from 4.
        merge_tables(f"{config['paths']['results_tables_path']}/{corpus_name}", config['parameters']['modes'], categories_ids = categories_ids)

    logger_experiment.info(f"Finished all experiments and summarized scores.")

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-c", "--config", help="filename with experiment configuration")
    argParser.add_argument("-g", "--gpu", help="set gpu id", default=0)

    args = argParser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)

    config['config_filepath'] = os.path.splitext(json_file.name)[0].split(os.sep)[-1]

    run(config, args.gpu)
