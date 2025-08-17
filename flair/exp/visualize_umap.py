import json
import numpy as np
import umap
import matplotlib.pyplot as plt
import argparse

from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from flair.exp.config import exp_data_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UMAP visualization of embeddings")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument(
        "--file",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = "tuned-embeddings.npy" if args.finetuned else "raw-embeddings.npy"
    X = np.load(exp_data_folder / output_name)
    labels = json.loads((exp_data_folder / "labels.json").read_text(encoding="utf-8"))

    if len(labels) != X.shape[0]:
        raise ValueError(f"Number of labels ({len(labels)}) does not match embeddings ({X.shape[0]}).")

    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=0)
    x_umap = reducer.fit_transform(X)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(["-".join(reversed(label.split("-"))) for label in labels])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_umap[:, 0], x_umap[:, 1], c=y, cmap="Spectral", s=5, alpha=0.7)

    classes = label_encoder.classes_
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=scatter.cmap(scatter.norm(i)),
            markersize=8,
            label="-".join(reversed(cls.split("-"))),
        )
        for i, cls in enumerate(classes)
    ]
    plt.legend(handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("UMAP Projection of Embeddings")

    if args.file:
        out_path = (exp_data_folder / output_name).with_suffix(".png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
