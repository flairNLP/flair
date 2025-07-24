import csv
import flair
import json
from pathlib import Path
import torch
from torch.utils.data.dataset import ConcatDataset
from typing import Dict, List, Optional, Union


def measure_noise_share(clean_corpus: flair.data.Corpus, noisy_corpus: flair.data.Corpus, label_type: str, splits: Union[str, list[str]] = ['dev', 'train']) -> float:

    # concatenate chosen splits to a single data object
    if type(splits) is str:
        splits = [splits]
    clean_data_splits = []
    noisy_data_splits = []
    for split in splits:
        if split == 'dev':
            clean_data_splits.append(clean_corpus.dev)
            noisy_data_splits.append(noisy_corpus.dev)
        elif split == 'test':
            clean_data_splits.append(clean_corpus.test)
            noisy_data_splits.append(noisy_corpus.test)
        elif split == 'train':
            clean_data_splits.append(clean_corpus.train)
            noisy_data_splits.append(noisy_corpus.train)
        else:
            # TODO: invalid split
            pass
    clean_data: ConcatDataset = ConcatDataset(clean_data_splits)
    noisy_data: ConcatDataset = ConcatDataset(noisy_data_splits)

    # count errors
    no_instances = 0
    no_errors = 0
    for clean_data_point, noisy_data_point in zip(flair.data._iter_dataset(clean_data), flair.data._iter_dataset(noisy_data)):
        clean_label = clean_data_point.get_label(label_type=label_type).value
        noisy_label = noisy_data_point.get_label(label_type=label_type).value
        if clean_label != noisy_label:
            no_errors += 1
        no_instances += 1

    # calculate noise share
    measured_noise_share = no_errors / no_instances

    return measured_noise_share


def write_corpus_to_csv(simulation_corpus: flair.data.Corpus, data_folder: Path, label_type: str, splits: Union[str, list[str]] = ['dev', 'train']) -> None:   #TODO: data_folder

    Path(data_folder).mkdir(parents=True, exist_ok=True)
    if type(splits) is str:
        splits = [splits]
    for split in splits:
        if split == 'dev':
            with open(f"{data_folder}/dev.csv", "w", newline="", encoding="utf-8") as dev_file:
                dev_data = []
                for data_point in flair.data._iter_dataset(simulation_corpus.dev):
                    dev_data.append([data_point.text, data_point.get_label(label_type=label_type).value])
                simulated_dev_data = csv.writer(dev_file, delimiter="\t")
                simulated_dev_data.writerows(dev_data)          
        elif split == 'test':
            with open(f"{data_folder}/test.csv", "w", newline="", encoding="utf-8") as test_file:
                test_data = []
                for data_point in flair.data._iter_dataset(simulation_corpus.test):
                    test_data.append([data_point.text, data_point.get_label(label_type=label_type).value])
                simulated_test_data = csv.writer(test_file, delimiter="\t")
                simulated_test_data.writerows(test_data)
        elif split == 'train':
            with open(f"{data_folder}/train.csv", "w", newline="", encoding="utf-8") as train_file:
                train_data = []
                for data_point in flair.data._iter_dataset(simulation_corpus.train):
                    train_data.append([data_point.text, data_point.get_label(label_type=label_type).value])
                simulated_train_data = csv.writer(train_file, delimiter="\t")
                simulated_train_data.writerows(train_data) 
        else:
            # TODO: invalid split
            pass


def error_statistics(clean_corpus: flair.data.Corpus, simulation_corpus: flair.data.Corpus, label_type: str, splits: Union[str, list[str]] = ['dev', 'train']) -> Dict:

    label_dict = clean_corpus.make_label_dictionary(label_type=label_type, min_count=0, add_unk=False, add_dev_test=True)
    labels = label_dict.get_items()

    error_statistics_pre = {"total": {"T": 0,
                                      "F": 0}}
    for label in labels:
        error_statistics_pre[label] = {"TP": 0,
                                      "TN": 0,
                                      "FP": 0,
                                      "FN": 0}
    
    # concatenate chosen splits to a single data object
    if type(splits) is str:
        splits = [splits]
    clean_data_splits = []
    simulation_data_splits = []
    for split in splits:
        if split == 'dev':
            clean_data_splits.append(clean_corpus.dev)
            simulation_data_splits.append(simulation_corpus.dev)
        elif split == 'test':
            clean_data_splits.append(clean_corpus.test)
            simulation_data_splits.append(simulation_corpus.test)
        elif split == 'train':
            clean_data_splits.append(clean_corpus.train)
            simulation_data_splits.append(simulation_corpus.train)
        else:
            # TODO: invalid split
            pass
    clean_data: ConcatDataset = ConcatDataset(clean_data_splits)
    simulation_data: ConcatDataset = ConcatDataset(simulation_data_splits)

    # count error types
    for clean_data_point, simulation_data_point in zip(flair.data._iter_dataset(clean_data), flair.data._iter_dataset(simulation_data)):
        clean_label = clean_data_point.get_label(label_type=label_type).value
        simulation_label = simulation_data_point.get_label(label_type=label_type).value

        if clean_label == simulation_label:
            error_statistics_pre["total"]["T"] += 1

            error_statistics_pre[simulation_label]["TP"] += 1
            for label in labels:
                if label != simulation_label:
                    error_statistics_pre[label]["TN"] += 1
        else:
            error_statistics_pre["total"]["F"] += 1

            error_statistics_pre[simulation_label]["FP"] += 1
            error_statistics_pre[clean_label]["FN"] += 1
            for label in labels:
                if label != simulation_label and label != clean_label:
                    error_statistics_pre[label]["TN"] += 1

    # compute error statistics
    error_statistics = {}
    for label in labels:
        error_statistics[label] = {"COUNT": error_statistics_pre[label]["TP"] + error_statistics_pre[label]["FN"],
                                   "PREC": error_statistics_pre[label]["TP"] / (error_statistics_pre[label]["TP"] + error_statistics_pre[label]["FP"]),
                                   "REC": error_statistics_pre[label]["TP"] / (error_statistics_pre[label]["TP"] + error_statistics_pre[label]["FN"])}
        error_statistics[label]["F1"] = 2 * error_statistics[label]["PREC"] * error_statistics[label]["REC"] / (error_statistics[label]["PREC"] + error_statistics[label]["REC"])

    sum_TP = 0
    sum_FP = 0
    sum_FN = 0
    for label in labels:
        sum_TP += error_statistics_pre[label]["TP"]
        sum_FP += error_statistics_pre[label]["FP"]
        sum_FN += error_statistics_pre[label]["FN"]

    error_statistics["total"] = {"COUNT": error_statistics_pre["total"]["T"] + error_statistics_pre["total"]["F"]}
    error_statistics["total"].update({"ACC": error_statistics_pre["total"]["T"] / error_statistics["total"]["COUNT"],
                                      "micro_PREC": sum_TP / (sum_TP + sum_FP),
                                      "micro_REC": sum_TP / (sum_TP + sum_FN)})
    error_statistics["total"]["micro_F1"] = 2 * error_statistics["total"]["micro_PREC"] * error_statistics["total"]["micro_REC"] / (error_statistics["total"]["micro_PREC"] + error_statistics["total"]["micro_REC"])

    return error_statistics


def write_specifications_file(dataset: str, label_type: str, noise_model: str, seed: int, target_noise_share: float, data_folder: Path, resulting_noise_share: float, error_statistics: Dict, noise_model_specs: Optional[Union[Dict, List, torch.Tensor]] = None) -> None:

    with open(f"{data_folder}/specs_{dataset}_{noise_model}_{seed}.txt", "w") as specs_file:
        specs_file.write(f"dataset: {dataset} \n")
        specs_file.write(f"label type: {label_type} \n")
        specs_file.write(f"noise model: {noise_model} \n")
        specs_file.write(f"seed: {seed} \n")
        specs_file.write(f"target noise share: {target_noise_share} \n")
        specs_file.write(f"resulting noise share: {resulting_noise_share} \n")
        if noise_model_specs:
            if isinstance(noise_model_specs, Dict):
                specs_file.write(f"noise model details: {json.dumps(noise_model_specs, indent=4)} \n")
            else:
                specs_file.write(f"noise model details: {noise_model_specs} \n")
        specs_file.write(f"error statistics: {json.dumps(error_statistics, indent=4)} \n")


def write_consistency_specs_file(dataset: str, label_type: str, noise_model: str, seed: int, noise_share: float, data_folder: Path, micro_f1, scores, detailed_results, classification_report):

    with open(f"{data_folder}/consistency_specs_{dataset}_{noise_model}_{seed}.txt", "w") as specs_file:
        specs_file.write(f"dataset: {dataset} \n")
        specs_file.write(f"label type: {label_type} \n")
        specs_file.write(f"noise model: {noise_model} \n")
        specs_file.write(f"seed: {seed} \n")
        specs_file.write(f"target noise share: {noise_share} \n")
        specs_file.write(f"micro F!: {micro_f1} \n")
        specs_file.write(f"scores: {json.dumps(scores, indent=4)} \n")
        specs_file.write(f"detailed results: {detailed_results} \n")
        specs_file.write(f"classification report: {json.dumps(classification_report, indent=4)} \n")


def write_stddev_file(dataset: str, label_type: str, noise_model: str, noise_share: float, data_folder: Path, results_F1: List[float], stddev: float):

    with open(f"{data_folder}/stddev_results_{dataset}_{noise_model}.txt", "w") as specs_file:
        specs_file.write(f"dataset: {dataset} \n")
        specs_file.write(f"label type: {label_type} \n")
        specs_file.write(f"noise model: {noise_model} \n")
        specs_file.write(f"target noise share: {noise_share} \n")
        specs_file.write(f"F1 scores: {results_F1} \n")
        specs_file.write(f"stddev: {stddev} \n")