import argparse

import flair
from flair.datasets import CONLL_03, FEWNERD, ONTONOTES, WNUT_17
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
    elif args.corpus == "fewnerd":
        corpus = FEWNERD(
            label_name_map={
                "location-GPE": "location geographical social political entity",
                "person-other": "person other",
                "organization-other": "organization other",
                "organization-company": "organization company",
                "person-artist/author": "person author artist",
                "person-athlete": "person athlete",
                "person-politician": "person politician",
                "building-other": "building other",
                "organization-sportsteam": "organization sportsteam",
                "organization-education": "organization eduction",
                "location-other": "location other",
                "other-biologything": "other biology",
                "location-road/railway/highway/transit": "location road railway highway transit",
                "person-actor": "person actor",
                "prodcut-other": "product other",
                "event-sportsevent": "event sportsevent",
                "organization-government/governmentagency": "organization government agency",
                "location-bodiesofwater": "location bodies of water",
                "organization-media/newspaper": "organization media newspaper",
                "art-music": "art music",
                "other-chemicalthing": "other chemical",
                "event-attack/battle/war/militaryconflict": "event attack war battle military conflict",
                "art-writtenart": "art written art",
                "other-award": "other award",
                "other-livingthing": "other living",
                "event-other": "event other",
                "art-film": "art film",
                "product-software": "product software",
                "organization-sportsleague": "organization sportsleague",
                "other-language": "other language",
                "other-disease": "other disease",
                "organization-showorganization": "organization show organization",
                "product-airplane": "product airplane",
                "other-astronomything": "other astronomy",
                "organization-religion": "organization religion",
                "product-car": "product car",
                "person-scholar": "person scholar",
                "other-currency": "other currency",
                "person-soldier": "person soldier",
                "location-mountain": "location mountain",
                "art-broadcastprogramm": "art broadcastprogramm",
                "location-island": "location island",
                "art-other": "art other",
                "person-director": "person director",
                "product-weapon": "product weapon",
                "other-god": "other god",
                "building-theater": "building theater",
                "other-law": "other law",
                "product-food": "product food",
                "other-medical": "other medical",
                "product-game": "product game",
                "location-park": "location park",
                "product-ship": "product ship",
                "building-sportsfacility": "building sportsfacility",
                "other-educationaldegree": "other educational degree",
                "building-airport": "building airport",
                "building-hospital": "building hospital",
                "product-train": "product train",
                "building-library": "building library",
                "building-hotel": "building hotel",
                "building-restaurant": "building restaurant",
                "event-disaster": "event disaster",
                "event-election": "event election",
                "event-protest": "event protest",
                "art-painting": "art painting",
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
    parser.add_argument("--corpus", type=str, default="fewnerd")
    parser.add_argument("--transformer", type=str, default="bert-base-cased")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)
