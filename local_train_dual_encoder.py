import argparse

import flair
from flair.datasets import CONLL_03, ONTONOTES, WNUT_17
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings
from flair.models import DualEncoder
from flair.trainers import ModelTrainer


def main(args):
    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    if args.corpus == "wnut_17":
        corpus = WNUT_17(
            label_name_map={
                "corporation": "corporation",
                "creative-work": "creative work",
                "group": "group",
                "location": "location",
                "person": "person",
                "product": "product",
            }
        )
    elif args.corpus == "conll_03":
        corpus = CONLL_03(
            base_path="data",
            column_format={0: "text", 1: "pos", 2: "chunk", 3: "ner"},
            label_name_map={"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "miscellaneous"},
        )
    elif args.corpus == "ontonotes":
        corpus = ONTONOTES(
            label_name_map={
                "CARDINAL": "cardinal",
                "DATE": "date",
                "EVENT": "event",
                "FAC": "facility",
                "GPE": "geographical social political entity",
                "LANGUAGE": "language",
                "LAW": "law",
                "LOC": "location",
                "MONEY": "money",
                "NORP": "nationality religion political",
                "ORDINAL": "ordinal",
                "ORG": "organization",
                "PERCENT": "percent",
                "PERSON": "person",
                "PRODUCT": "product",
                "QUANTITY": "quantity",
                "TIME": "time",
                "WORK_OF_ART": "work of art",
            }
        )
    else:
        raise Exception("no valid corpus.")

    tag_type = "ner"
    label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)

    token_encoder = TransformerWordEmbeddings(args.transformer)
    label_encoder = TransformerDocumentEmbeddings(args.transformer)

    model = DualEncoder(
        token_encoder=token_encoder, label_encoder=label_encoder, tag_dictionary=label_dictionary, tag_type=tag_type
    )

    trainer = ModelTrainer(model, corpus)

    trainer.fine_tune(
        f"{args.cache_path}/{args.transformer}_{args.corpus}_{args.lr}_{args.seed}",
        learning_rate=args.lr,
        mini_batch_size=args.bs,
        mini_batch_chunk_size=args.mbs,
        max_epochs=args.epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--cuda_device", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--cache_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon/flair-models/pretrained-dual-encoder"
    )
    parser.add_argument("--corpus", type=str, default="conll_03")
    parser.add_argument("--transformer", type=str, default="bert-base-cased")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    main(args)
