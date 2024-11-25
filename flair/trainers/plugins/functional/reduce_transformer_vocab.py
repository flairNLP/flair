import logging
from pathlib import Path

from transformer_smaller_training_vocab import reduce_train_vocab

from flair.embeddings import Embeddings, StackedEmbeddings, TransformerEmbeddings
from flair.models import FewshotClassifier
from flair.nn import Model
from flair.nn.model import ReduceTransformerVocabMixin
from flair.trainers.plugins import TrainerPlugin

log = logging.getLogger("flair")


class ReduceTransformerVocabPlugin(TrainerPlugin):
    def __init__(self, base_path: Path, save_optimizer_state: bool):
        super().__init__()
        self.base_path = base_path
        self.save_optimizer_state = save_optimizer_state
        self.disabled = False

    @TrainerPlugin.hook("after_setup")
    def register_transformer_smaller_training_vocab(self, **kw):
        if not isinstance(self.model, ReduceTransformerVocabMixin):
            log.warning("Cannot reduce the transformer vocab: model is not supported.")
            self.disabled = True
            return

        embeddings = get_transformer_embeddings(self.model)
        if not embeddings:
            self.disabled = True
            log.warning("Cannot reduce the transformer vocab: no transformer embeddings found.")

        max_context_length = max(emb.context_length for emb in embeddings)
        respect_document_boundaries = all(emb.respect_document_boundaries for emb in embeddings)

        tokens = list(
            filter(None, self.model.get_used_tokens(self.corpus, max_context_length, respect_document_boundaries))
        )
        for emb in embeddings:
            self.trainer.context_stack.enter_context(
                reduce_train_vocab(
                    model=emb.model, tokenizer=emb.tokenizer, texts=tokens, optimizer=self.trainer.optimizer
                )
            )

    @TrainerPlugin.hook("after_training")
    def save_model_at_the_end(self, **kw):
        # saves the model with full vocab as checkpoints etc were created with reduced vocab.
        if self.disabled:
            return

        if (self.base_path / "best-model.pt").exists():
            self.model.save(self.base_path / "best-model.pt", checkpoint=self.save_optimizer_state)
        elif (self.base_path / "final-model.pt").exists():
            self.model.save(self.base_path / "final-model.pt", checkpoint=self.save_optimizer_state)

    @property
    def attach_to_all_processes(self) -> bool:
        return False


def get_transformer_embeddings(model: Model) -> list[TransformerEmbeddings]:
    embeddings = model.tars_embeddings if isinstance(model, FewshotClassifier) else getattr(model, "embeddings", None)

    if embeddings is None:
        log.warning(f"Could not extract embeddings of Model of type {type(model)}")
        return []

    transformer_embeddings = set()

    def scan_embeddings(emb: Embeddings):
        if isinstance(emb, StackedEmbeddings):
            for sub_emb in emb.embeddings:
                scan_embeddings(sub_emb)
        if isinstance(emb, TransformerEmbeddings):
            transformer_embeddings.add(emb)

    scan_embeddings(embeddings)

    return list(transformer_embeddings)
