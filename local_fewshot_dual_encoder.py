import argparse

import flair
from flair.datasets import CONLL_03, WNUT_17, ColumnCorpus
from flair.models import DualEncoder
from flair.trainers import ModelTrainer


def main(args):
    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    for support_set_id in range(5):
        if args.corpus == "wnut_17":
            few_shot_corpus = ColumnCorpus(
                data_folder=f"data/fewshot/wnut17/{args.k}shot/",
                train_file=f"{support_set_id}.txt",
                column_format={0: "text", 1: "ner"},
                sample_missing_splits=False,
                label_name_map={
                    "corporation": "corporation",
                    "creative-work": "creative work",
                    "group": "group",
                    "location": "location",
                    "person": "person",
                    "product": "product",
                },
            )

            full_corpus = WNUT_17(
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
            few_shot_corpus = ColumnCorpus(
                data_folder=f"data/fewshot/conll03/{args.k}shot/",
                train_file=f"{support_set_id}.txt",
                column_format={0: "text", 1: "ner"},
                sample_missing_splits=False,
                label_name_map={"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "miscellaneous"},
            )

            full_corpus = CONLL_03(
                base_path="data",
                column_format={0: "text", 1: "pos", 2: "chunk", 3: "ner"},
                label_name_map={"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "miscellaneous"},
            )
        else:
            raise Exception("no valid corpus.")

        tag_type = "ner"
        label_dictionary = few_shot_corpus.make_label_dictionary(tag_type, add_unk=False)
        # force spans == true, there is one split containing only B-*'s
        label_dictionary.span_labels = True

        model = DualEncoder.load(args.pretrained_model)
        model._init_verbalizers_and_tag_dictionary(tag_dictionary=label_dictionary)

        trainer = ModelTrainer(model, few_shot_corpus)

        trainer.fine_tune(
            f"{args.cache_path}/{args.transformer}_{args.corpus}_{args.lr}_{args.seed}/{args.k}shot_{support_set_id}",
            learning_rate=args.lr,
            mini_batch_size=args.bs,
            mini_batch_chunk_size=args.mbs,
            max_epochs=args.epochs,
            save_final_model=False,
        )

        result = model.evaluate(
            data_points=full_corpus.test,
            gold_label_type=tag_type,
            out_path=f"{args.cache_path}/{args.transformer}_{args.corpus}_{args.lr}_{args.seed}/{args.k}shot_{support_set_id}/predictions.txt",
        )
        with open(
            f"{args.cache_path}/{args.transformer}_{args.corpus}_{args.lr}_{args.seed}/{args.k}shot_{support_set_id}/result.txt",
            "w",
        ) as f:
            f.writelines(result.detailed_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--cuda_device", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="/glusterfs/dfs-gfs-dist/goldejon/flair-models/pretrained-dual-encoder/bert-base-cased_ontonotes_1e-05_123/final-model.pt",
    )
    parser.add_argument(
        "--cache_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon/flair-models/fewshot-dual-encoder"
    )
    parser.add_argument("--corpus", type=str, default="conll_03")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--transformer", type=str, default="bert-base-cased")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    main(args)
