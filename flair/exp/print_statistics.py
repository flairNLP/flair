import argparse
import json

import numpy as np

from flair.exp.config import get_embeddings_file, exp_data_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--o_tuned", action="store_true")
    return parser.parse_args()


def print_stats(embeddings_vector: np.ndarray) -> None:
    mean = np.mean(embeddings_vector, axis=0)
    covariance_matrix = np.cov(embeddings_vector, rowvar=False)
    scatter = ((embeddings_vector - mean) ** 2).mean(axis=0).sum()

    print("mean-norm: ", np.linalg.norm(mean))
    print("mean-mean: ", mean.mean())
    print("mean-abs.mean: ", np.abs(mean).mean())
    print("covar-trace: ", np.linalg.trace(covariance_matrix))
    print("covar-largest-eigvalue: ", max(np.linalg.eig(covariance_matrix).eigenvalues))
    print("scatter: ", scatter)
    print()


def main() -> None:
    args = parse_args()

    output_name = get_embeddings_file(args)
    embeddings_vector = np.load(exp_data_folder / output_name)
    labels = json.loads((exp_data_folder / "labels.json").read_text(encoding="utf-8"))
    label_names = sorted(set(labels), key=lambda l: "-".join(reversed(l.split("-"))) if l != "O" else "")
    print(label_names)
    label_np = np.array([label_names.index(l) for l in labels])

    print("Global:")
    print_stats(embeddings_vector)

    print("O:")
    print_stats(embeddings_vector[label_np == 0, :])

    print("Other than O:")
    print_stats(embeddings_vector[label_np != 0, :])


if __name__ == "__main__":
    main()
