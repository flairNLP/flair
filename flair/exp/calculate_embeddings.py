import argparse
import json
from itertools import batched

import numpy as np
import torch
from tqdm import tqdm

from flair.exp.config import (
    load_dataset,
    load_finetuned_embeddings,
    load_embeddings,
    exp_data_folder,
    get_embeddings_file,
    load_finetuned_o_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--o_tuned", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset()

    embeddings = (
        load_finetuned_embeddings()
        if args.finetuned
        else load_embeddings() if not args.o_tuned else load_finetuned_o_embeddings()
    )
    embeddings.fine_tune = False

    for batch in batched(tqdm(ds, desc="Embedding sentences"), args.batch_size):
        embeddings.embed(list(batch))

    all_embeddings = []
    all_labels = []
    for sentence in ds:
        all_embeddings.extend(token.embedding for token in sentence)
        labels = ["O"] * len(sentence)
        for entity in sentence.get_spans("ner"):
            label = entity.labels[0].value
            if len(entity.tokens) == 1:
                labels[entity.tokens[0].idx - 1] = f"S-{label}"
            else:
                for token in entity.tokens:
                    labels[token.idx - 1] = f"I-{label}"
                labels[entity.tokens[0].idx - 1] = f"B-{label}"
                labels[entity.tokens[-1].idx - 1] = f"E-{label}"
        all_labels.extend(labels)
        sentence.clear_embeddings()
    embeddings_vector = torch.stack(all_embeddings, 0).cpu().numpy()
    del embeddings
    output_name = get_embeddings_file(args)
    (exp_data_folder / "labels.json").write_text(json.dumps(all_labels, indent=4), encoding="utf-8")
    np.save(exp_data_folder / output_name, embeddings_vector)


if __name__ == "__main__":
    main()
