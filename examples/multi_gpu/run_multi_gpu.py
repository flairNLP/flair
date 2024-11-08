import torch

import flair
from flair.datasets import IMDB
from flair.distributed_utils import launch_distributed
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


def main(multi_gpu):
    # Note: Multi-GPU can affect corpus loading
    # This code will run multiple times -- each GPU gets its own process and each process runs this code. We need to
    # ensure that the corpus has the same elements and order on all processes, despite sampling. We do that by using
    # the same seed on all processes.
    flair.set_seed(1336)

    corpus = IMDB()
    corpus.downsample(0.01)
    label_type = "sentiment"
    label_dictionary = corpus.make_label_dictionary(label_type)

    embeddings = TransformerDocumentEmbeddings(model="distilbert-base-uncased")
    model = TextClassifier(embeddings, label_type, label_dictionary=label_dictionary)

    # Note: Multi-GPU can affect choice of batch size.
    # In order to compare batch updates fairly between single and multi-GPU training, we should:
    #   1) Step the optimizer after the same number of examples to achieve com
    #   2) Process the same number of examples in each forward pass
    mini_batch_chunk_size = 32  # Make this as large as possible without running out of GPU-memory to pack device
    num_devices_when_distributing = max(torch.cuda.device_count(), 1)
    mini_batch_size = mini_batch_chunk_size if multi_gpu else mini_batch_chunk_size * num_devices_when_distributing
    # e.g. Suppose your machine has 2 GPUs. If multi_gpu=False, the first gpu will process 32 examples, then the
    # first gpu will process another 32 examples, then the optimizer will step. If multi_gpu=True, each gpu will
    # process 32 examples at the same time, then the optimizer will step.

    trainer = ModelTrainer(model, corpus)
    trainer.train(
        "resources/taggers/multi-gpu",
        multi_gpu=multi_gpu,  # Required for multi-gpu
        max_epochs=2,
        mini_batch_chunk_size=mini_batch_chunk_size,
        mini_batch_size=mini_batch_size,
    )


if __name__ == "__main__":
    """Minimal example demonstrating how to train a model on multiple GPUs."""
    multi_gpu = True

    if multi_gpu:
        launch_distributed(main, multi_gpu)  # Required for multi-gpu
    else:
        main(multi_gpu)
