import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass, field

import torch
from transformers import HfArgumentParser

import flair
from flair import set_seed
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

logger = logging.getLogger("flair")
logger.setLevel(level="INFO")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    layers: str = field(default="-1", metadata={"help": "Layers to be fine-tuned."})
    subtoken_pooling: str = field(
        default="first",
        metadata={"help": "Subtoken pooling strategy used for fine-tuned."},
    )
    hidden_size: int = field(default=256, metadata={"help": "Hidden size for NER model."})
    use_crf: bool = field(default=False, metadata={"help": "Whether to use a CRF on-top or not."})


@dataclass
class TrainingArguments:
    num_epochs: int = field(default=10, metadata={"help": "The number of training epochs."})
    batch_size: int = field(default=8, metadata={"help": "Batch size used for training."})
    mini_batch_chunk_size: int = field(
        default=1,
        metadata={"help": "If smaller than batch size, batches will be chunked."},
    )
    learning_rate: float = field(default=5e-05, metadata={"help": "Learning rate"})
    seed: int = field(default=42, metadata={"help": "Seed used for reproducible fine-tuning results."})
    device: str = field(default="cuda:0", metadata={"help": "CUDA device string."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for optimizer."})
    embeddings_storage_mode: str = field(default="none", metadata={"help": "Defines embedding storage method."})


@dataclass
class FlertArguments:
    context_size: int = field(default=0, metadata={"help": "Context size when using FLERT approach."})
    respect_document_boundaries: bool = field(
        default=False,
        metadata={"help": "Whether to respect document boundaries or not when using FLERT."},
    )


@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "Flair NER dataset name."})
    dataset_arguments: str = field(default="", metadata={"help": "Dataset arguments for Flair NER dataset."})
    output_dir: str = field(
        default="resources/taggers/ner",
        metadata={"help": "Defines output directory for final fine-tuned model."},
    )


def get_flair_corpus(data_args):
    ner_task_mapping = {}

    for name, obj in inspect.getmembers(flair.datasets.sequence_labeling):
        if inspect.isclass(obj):
            if name.startswith("NER") or name.startswith("CONLL") or name.startswith("WNUT"):
                ner_task_mapping[name] = obj

    dataset_args = {}
    dataset_name = data_args.dataset_name

    if data_args.dataset_arguments:
        dataset_args = json.loads(data_args.dataset_arguments)

    if dataset_name not in ner_task_mapping:
        raise ValueError(f"Dataset name {dataset_name} is not a valid Flair datasets name!")

    return ner_task_mapping[dataset_name](**dataset_args)


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, FlertArguments, DataArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (
            model_args,
            training_args,
            flert_args,
            data_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            training_args,
            flert_args,
            data_args,
        ) = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    flair.device = training_args.device

    corpus = get_flair_corpus(data_args)

    logger.info(corpus)

    tag_type: str = "ner"
    tag_dictionary = corpus.make_label_dictionary(tag_type)
    logger.info(tag_dictionary)

    embeddings = TransformerWordEmbeddings(
        model=model_args.model_name_or_path,
        layers=model_args.layers,
        subtoken_pooling=model_args.subtoken_pooling,
        fine_tune=True,
        use_context=flert_args.context_size,
        respect_document_boundaries=flert_args.respect_document_boundaries,
    )

    tagger = SequenceTagger(
        hidden_size=model_args.hidden_size,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=model_args.use_crf,
        use_rnn=False,
        reproject_embeddings=False,
    )

    trainer = ModelTrainer(tagger, corpus)

    trainer.fine_tune(
        data_args.output_dir,
        learning_rate=training_args.learning_rate,
        mini_batch_size=training_args.batch_size,
        mini_batch_chunk_size=training_args.mini_batch_chunk_size,
        max_epochs=training_args.num_epochs,
        embeddings_storage_mode=training_args.embeddings_storage_mode,
        weight_decay=training_args.weight_decay,
    )

    torch.save(model_args, os.path.join(data_args.output_dir, "model_args.bin"))
    torch.save(training_args, os.path.join(data_args.output_dir, "training_args.bin"))

    # finally, print model card for information
    tagger.print_model_card()


if __name__ == "__main__":
    main()
