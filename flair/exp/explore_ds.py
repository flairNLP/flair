from collections import defaultdict

from flair.exp.config import load_dataset


def main() -> None:
    ds = load_dataset()
    token_count = 0
    count_per_entity = defaultdict(int)
    count_per_label = defaultdict(int)

    for sent in ds:
        token_count += len(sent)
        o_count = len(sent)
        for entity in sent.get_spans("ner"):
            label = entity.labels[0].value
            count_per_entity[label] += len(entity.tokens)
            o_count -= len(entity.tokens)
            if len(entity.tokens) > 1:
                count_per_label[f"I-{label}"] += len(entity.tokens) - 2
                count_per_label[f"B-{label}"] += 1
                count_per_label[f"E-{label}"] += 1
            else:
                count_per_label[f"S-{label}"] += 1
        count_per_label["O"] += o_count
    print(f"Token count: {token_count}")
    print(f"Count per Entity: {dict(count_per_entity)}")
    print(f"Count per Label: {dict(count_per_label)}")


if __name__ == "__main__":
    main()
