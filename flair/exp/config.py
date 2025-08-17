import argparse
from pathlib import Path

from flair.data import Sentence, Corpus, Dictionary
from flair.datasets import CONLL_03
from flair.embeddings import TokenEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger


exp_data_folder = Path(__file__).parents[2] / "exp"
exp_data_folder.mkdir(exist_ok=True, parents=True)

model_finetune_path = exp_data_folder / "finetuned-model.pt"
model_finetune_o_path = exp_data_folder / "finetuned-o-model.pt"


def load_train_corpus() -> Corpus:
    return CONLL_03()


def load_dataset() -> list[Sentence]:
    return list(load_train_corpus().test)


def load_embeddings() -> TokenEmbeddings:
    return TransformerWordEmbeddings(
        "xlm-roberta-base",
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=False,
    )


def load_finetuned_embeddings() -> TokenEmbeddings:
    emb = SequenceTagger.load(model_finetune_path).embeddings
    emb.fine_tune = False
    return emb


def load_finetuned_o_embeddings() -> TokenEmbeddings:
    emb = SequenceTagger.load(model_finetune_o_path).embeddings
    emb.fine_tune = False
    return emb


def create_sequence_tagger_for_o_train(tag_dictionary: Dictionary) -> SequenceTagger:
    emb = load_embeddings()
    return SequenceTagger(
        emb,
        tag_dictionary,
        "ner",
        use_crf=True,
        use_rnn=False,
        reproject_embeddings=False,
        o_count=20,
    )


def create_sequence_tagger_for_train(tag_dictionary: Dictionary) -> SequenceTagger:
    emb = load_embeddings()
    return SequenceTagger(
        emb,
        tag_dictionary,
        "ner",
        use_crf=True,
        use_rnn=False,
        reproject_embeddings=False,
    )


def get_embeddings_file(args: argparse.Namespace) -> str:
    return (
        "tuned-embeddings.npy"
        if args.finetuned
        else "raw-embeddings.npy" if not args.o_tuned else "otuned-embeddings.npy"
    )
