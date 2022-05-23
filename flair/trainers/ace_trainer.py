import collections
import copy
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Type, Union, cast

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD, Optimizer
from tqdm import tqdm

import flair
from flair.data import Corpus, MultiCorpus, Sentence
from flair.datasets import DataLoader
from flair.embeddings import Embeddings, StackedEmbeddings, TokenEmbeddings
from flair.models import (
    DependencyParser,
    EntityLinker,
    Lemmatizer,
    RelationExtractor,
    SequenceTagger,
    TextPairClassifier,
)
from flair.nn import Model
from flair.trainers.trainer import ModelTrainer
from flair.training_utils import add_file_handler, log_line, store_embeddings

_label_type_model_mapping: Dict[str, Type[Model]] = {
    "dependency": DependencyParser,
    "ner": SequenceTagger,
    "pos": SequenceTagger,
    "nel": EntityLinker,
    "lemma": Lemmatizer,
    "pair": TextPairClassifier,
    "relation": RelationExtractor,
    "class": RelationExtractor,
}

log = logging.getLogger("flair")


class AceEmbeddings(Embeddings[Sentence]):
    def __init__(self, embeddings: List[Embeddings]):
        # this is not a torch.nn.ModuleList on purpose, so the embeddings won't be added as parameters to the model.
        self.embeddings = embeddings
        self.active = torch.ones((len(embeddings),), device=flair.device, dtype=torch.bool)
        self.configurations: Dict[str, torch.Tensor] = {}
        self.rewards: Dict[str, float] = {}
        super().__init__()

    @property
    def embedding_length(self) -> int:
        current_embedding_length = 0
        for active_flag, embedding in zip(self.active, self.embeddings):
            if active_flag:
                current_embedding_length += embedding.embedding_length
        return current_embedding_length

    @property
    def embedding_type(self) -> str:
        return self.embeddings[0].embedding_type

    @property
    def active_key(self) -> str:
        return self.state_to_key(self.active)

    def get_names(self):
        names = []
        for active_flag, embedding in zip(self.active, self.embeddings):
            if active_flag:
                names.append(embedding.name)
        return names

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        # do nothing, as we expect the embeddings to be already stored on the cpu
        pass

    @staticmethod
    def state_to_key(state: torch.Tensor) -> str:
        return "".join([str(int(b)) for b in state])

    def create_log_prop_parameter(self) -> Variable:
        # The parameter won't be part of the embedding, so it won't be added at the plain trainings.
        # it will be handled in the training loop.
        return Variable(torch.zeros(len(self.embeddings), device=flair.device), requires_grad=True)

    def compute_loss(self, score: float, discount: float, log_prop_weights: torch.Tensor) -> torch.Tensor:
        reward_per_embedding = torch.zeros(len(self.embeddings), device=flair.device)
        one_prob = torch.sigmoid(log_prop_weights)
        m = torch.distributions.Bernoulli(one_prob)
        log_prop = m.log_prob(self.active)

        for key in self.configurations.keys():
            dist = torch.logical_xor(self.active, self.configurations[key])
            reward = discount ** (dist.sum() - 1) * (score - self.rewards[key])
            reward_per_embedding += dist * reward

        return -(log_prop * reward_per_embedding).sum()

    def save_configuration(self, score: float):
        key = self.state_to_key(self.active)
        self.configurations[key] = self.active
        self.rewards[key] = score
        log.info(f"Config with embeddings: {self.get_names()} achieved a score of {score}")

    def best_to_stacked(self, log_prop_weights: torch.Tensor):
        self.active = log_prop_weights >= 0.0
        for emb in self.embeddings:
            emb.to(flair.device)

        log.info(f"Final model with embeddings: {self.get_names()}")
        return StackedEmbeddings(
            cast(List[TokenEmbeddings], [emb for (can_use, emb) in zip(self.active, self.embeddings)])
        )

    def sample_config(self, log_prop_weights: torch.Tensor):
        one_prob = torch.sigmoid(log_prop_weights)
        m = torch.distributions.Bernoulli(one_prob)
        log.info(f"Embedding probabilities: {one_prob}")
        v = m.sample()
        sample_tries = 1000
        while v.sum().item() == 0 or self.state_to_key(v) in self.configurations:
            sample_tries -= 1
            if sample_tries == -1:
                break
            v = m.sample()
        self.active = v
        if sample_tries == -1:
            log.warning(
                "Bernoulli sampling won't work due to too good coverage of likely solutions."
                "Using exhaustive prop instead."
            )
            all_possibilities = []
            props = []
            n_embeddings = len(self.embeddings)
            for comb in torch.arange(1, 2**n_embeddings):
                mask = 2 ** torch.arange(n_embeddings - 1, -1, -1, device=flair.device, dtype=torch.int)
                val = comb.unsqueeze(-1).bitwise_and(mask).ne(0)
                if self.state_to_key(val) in self.configurations:
                    continue
                all_possibilities.append(val)
                p = torch.exp(m.log_prob(val)).item()
                props.append(p)
            np_props = np.array(props)
            np_props /= np_props.sum()
            idx = np.random.choice(np.arange(len(all_possibilities)), p=np_props)
            self.active = all_possibilities[idx]
        log.info(f"Set new configuration, embeddings are: {self.get_names()}")

    def adjust_init_state_dict(
        self, init: Dict[str, torch.Tensor], size: Dict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        """Take the state dict of a full embedding and remove the unused embeddings."""
        assert sorted(init.keys()) == sorted([k for k in size.keys() if "list_embedding_" not in k])
        result: OrderedDict[str, torch.Tensor] = collections.OrderedDict()
        total_embedding_size = sum(emb.embedding_length for emb in self.embeddings)
        curr_embedding_size = self.embedding_length
        for k in init.keys():
            if init[k].size() == size[k].size():
                result[k] = init[k]
                continue

            assert init[k].dim() == size[k].dim()
            reduced = init[k].clone()
            target_size = size[k].size()
            for i in range(size[k].dim()):
                if reduced.size(i) == target_size[i]:
                    continue
                iters = reduced.size(i) // total_embedding_size
                assert iters == target_size[i] // curr_embedding_size
                offset = 0
                if i != 0:
                    reduced = reduced.transpose(0, i)
                for _ in range(iters):
                    for is_current_active, embedding in zip(self.active, self.embeddings):
                        if not is_current_active:
                            reduced = torch.cat(
                                [reduced[:offset], reduced[offset + embedding.embedding_length :]], dim=0
                            )
                        else:
                            offset += embedding.embedding_length
                if i != 0:
                    reduced = reduced.transpose(0, i)

                assert reduced.size(i) == target_size[i]
            result[k] = reduced

        return result

    def update_state_dict(
        self, curr: Dict[str, torch.Tensor], full: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Take the updated state dict of a r embedding and put it back to the full state."""
        assert sorted(curr.keys()) == sorted(full.keys())
        result: Dict[str, torch.Tensor] = {}
        curr_embedding_size = self.embedding_length
        total_embedding_size = sum(emb.embedding_length for emb in self.embeddings)
        for k in curr.keys():
            if curr[k].size() == full[k].size():
                result[k] = curr[k]
                continue

            assert curr[k].dim() == full[k].dim()
            r = full[k].clone()
            small = curr[k]

            for i in range(r.dim()):
                if r.size(i) == small.size(i):
                    continue
                iters = r.size(i) // total_embedding_size
                assert iters == small.size(i) // curr_embedding_size
                curr_offset = 0
                target_offset = 0
                if i != 0:
                    r = r.transpose(0, i)
                    small = small.transpose(0, i)

                for _ in range(iters):
                    for is_current_active, embedding in zip(self.active, self.embeddings):
                        if is_current_active:
                            r[target_offset : target_offset + embedding.embedding_length] = small[
                                curr_offset : curr_offset + embedding.embedding_length
                            ]
                            curr_offset += embedding.embedding_length
                        target_offset += embedding.embedding_length
                if i != 0:
                    r = r.transpose(0, i)
                    small = small.transpose(0, i)

            result[k] = r

        return result


class AceTrainer:
    def __init__(
        self,
        corpus: Corpus,
        embeddings: List[Embeddings],
        model_args: Dict[str, Any],
        model_type: Union[str, Type[Model]] = None,
    ):
        if model_type is None:
            label_type: Optional[str] = model_args.get("label_type", model_args.get("tag_type", None))
            if label_type is None:
                raise ValueError("ModelType is not set and could not infer it via 'label_type' & 'tag_type'")
            model_type = label_type

        if isinstance(model_type, type):
            self.model_type = model_type
        else:
            self.model_type = _label_type_model_mapping[model_type]

        self.embeddings_parameter = [
            k for k in inspect.signature(self.model_type).parameters.keys() if "embedding" in k
        ][0]
        self.corpus = corpus
        self.embeddings = embeddings
        self.model_args = model_args

    def create_model_with_embeddings(self, ace_embedding: Union[AceEmbeddings, StackedEmbeddings]) -> Model:
        args = copy.deepcopy(self.model_args)
        args[self.embeddings_parameter] = ace_embedding
        return self.model_type(**args)

    def run_episode(
        self,
        base_path: Path,
        ace_embeddings: AceEmbeddings,
        train_args: Dict[str, Any],
        init_state_dict: Dict[str, torch.Tensor] = None,
    ) -> Tuple[bool, float, Dict[str, torch.Tensor]]:
        model = self.create_model_with_embeddings(ace_embeddings)
        if init_state_dict is not None:
            model.load_state_dict(ace_embeddings.adjust_init_state_dict(init=init_state_dict, size=model.state_dict()))

        trainer = ModelTrainer(model, self.corpus)
        training_results = trainer.train(
            base_path=base_path,
            train_with_dev=False,
            embeddings_storage_mode="cpu",
            param_selection_mode=True,
            test_embedding_storage_mode="cpu",
            **train_args,
        )
        if init_state_dict is not None:
            updated_state_dict = ace_embeddings.update_state_dict(curr=model.state_dict(), full=init_state_dict)
        else:
            updated_state_dict = model.state_dict()
        del model
        return training_results["did_stop_per_user"], training_results["dev_score_history"][-1], updated_state_dict

    def train(
        self,
        base_path: Union[Path, str],
        inner_train_args: Dict[str, Any],
        controller_learning_rate: float = 0.1,
        controller_optimizer: Type[Optimizer] = SGD,
        controller_loss_discount: float = 0.5,
        controller_momentum: float = 0.0,
        max_episodes: int = 30,
        embedding_batch_size: int = 1,
        create_file_logs: bool = True,
    ):
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True, parents=True)

        for emb in self.embeddings:
            emb.eval()
            emb.fine_tune = False  # type: ignore
            emb.static_embeddings = True
            emb.cpu()

        for emb in self.embeddings:
            all_data_loader = DataLoader(self.corpus.get_all_sentences(), batch_size=embedding_batch_size)
            emb.to(flair.device)
            for batch in tqdm(all_data_loader, desc=f"Computing Embeddings for ACE {emb.name}"):
                emb.embed(batch)
                store_embeddings(batch, "cpu")
            emb.cpu()
            if flair.device.type != "cpu":
                torch.cuda.empty_cache()

        ace_embeddings = AceEmbeddings(self.embeddings)
        log_prop = ace_embeddings.create_log_prop_parameter()
        state_dict = None
        log_handler = None
        prop_history: List[torch.Tensor] = []

        try:
            if create_file_logs:
                log_handler = add_file_handler(log, base_path / "training.log")
            else:
                log_handler = None

            if max_episodes > 2 ** len(self.embeddings) - 1:
                log.warning(f"More episodes than possible, shrinking amount down to {2 ** len(self.embeddings) - 1}")
                max_episodes = 2 ** len(self.embeddings) - 1

            log.info(f"Episode 1 / {max_episodes}: full embeddings")
            cancel, score, state_dict = self.run_episode(base_path / "full", ace_embeddings, inner_train_args)

            ace_embeddings.save_configuration(score)

            optim = controller_optimizer(params=[log_prop], lr=controller_learning_rate, momentum=controller_momentum)  # type: ignore

            for episode in range(1, max_episodes):
                if cancel:
                    break

                ace_embeddings.sample_config(log_prop)
                log.info(f"Episode {episode + 1} / {max_episodes}")
                cancel, score, state_dict = self.run_episode(
                    base_path / ace_embeddings.active_key, ace_embeddings, inner_train_args, init_state_dict=state_dict
                )
                optim.zero_grad()
                loss = ace_embeddings.compute_loss(score, controller_loss_discount, log_prop)
                loss.backward()
                optim.step()
                ace_embeddings.save_configuration(score)
                prop_history.append(torch.sigmoid(log_prop).detach().cpu().clone())
        except KeyboardInterrupt:
            cancel = True
        except Exception:
            if log_handler is not None:
                log_handler.close()
                log.removeHandler(log_handler)
            raise

        if cancel:
            log_line(log)
            log.info("Exiting ACE from training early.")

        log.info("Creating final Model")

        final_model = self.create_model_with_embeddings(ace_embeddings.best_to_stacked(log_prop))
        if state_dict is not None:
            # non strict state loading, as embeddings won't have any keys.
            final_model.load_state_dict(
                ace_embeddings.adjust_init_state_dict(init=state_dict, size=final_model.state_dict()), strict=False
            )
        final_model.eval()
        final_model.save(base_path / "best-ace-model.pt")

        if self.corpus.test is not None:
            log.info("Testing using final model ...")
            test_results = final_model.evaluate(
                self.corpus.test,
                gold_label_type=final_model.label_type,
                mini_batch_size=embedding_batch_size,
                out_path=base_path / "test.tsv",
                embedding_storage_mode="none",
                main_evaluation_metric=inner_train_args.get("main_evaluation_metric", ("micro avg", "f1-score")),
                gold_label_dictionary=inner_train_args.get("gold_label_dictionary_for_eval"),
                exclude_labels=inner_train_args.get("exclude_labels", []),
            )
            log.info(test_results.log_line)
            log.info(test_results.detailed_results)
            log_line(log)

            if isinstance(self.corpus, MultiCorpus):
                for subcorpus in self.corpus.corpora:
                    log_line(log)
                    if subcorpus.test:
                        subcorpus_results = final_model.evaluate(
                            subcorpus.test,
                            gold_label_type=final_model.label_type,
                            mini_batch_size=embedding_batch_size,
                            out_path=base_path / f"{subcorpus.name}-test.tsv",
                            embedding_storage_mode="none",
                            main_evaluation_metric=inner_train_args.get(
                                "main_evaluation_metric", ("micro avg", "f1-score")
                            ),
                            gold_label_dictionary=inner_train_args.get("gold_label_dictionary_for_eval"),
                            exclude_labels=inner_train_args.get("exclude_labels", []),
                        )
                        log.info(subcorpus.name)
                        log.info(subcorpus_results.log_line)

            final_score = test_results.main_score
        else:
            final_score = 0
            log.info("Test data not provided setting final score to 0")

        if log_handler is not None:
            log_handler.close()
            log.removeHandler(log_handler)

        return final_model, {
            "test_score": final_score,
            "tried_components": list(ace_embeddings.configurations.keys()),
            "score_per_component": list(ace_embeddings.rewards.values()),
            "did_stop_per_user": cancel,
            "final_probabilities": torch.sigmoid(log_prop),
            "probability_history": prop_history,
        }
