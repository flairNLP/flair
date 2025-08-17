import argparse

from flair.exp.config import (
    create_sequence_tagger_for_o_train,
    create_sequence_tagger_for_train,
    load_train_corpus,
    model_finetune_o_path,
    model_finetune_path,
)
from flair.trainers import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--o_tags", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus = load_train_corpus()
    tag_dictionary = corpus.make_label_dictionary("ner", add_unk=False)
    tagger = (
        create_sequence_tagger_for_o_train(tag_dictionary)
        if args.o_tags
        else create_sequence_tagger_for_train(tag_dictionary)
    )

    trainer = ModelTrainer(tagger, corpus)

    trainer.fine_tune(
        "training",
        reduce_transformer_vocab=True,
        max_epochs=10,
    )
    tagger.save(model_finetune_o_path if args.o_tags else model_finetune_path)


if __name__ == "__main__":
    main()
