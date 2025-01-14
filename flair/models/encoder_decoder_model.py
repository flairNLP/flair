import logging
import math
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import inspect
import ast

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    GenerationMixin,
    PreTrainedModel,
)
from transformers.modeling_outputs import Seq2SeqLMOutput

import flair
from flair.data import DataPoint
from flair.datasets import FlairDatapointDataset
from flair.nn import Model
from flair.training_utils import Result
from flair.embeddings.base import load_embeddings


logger = logging.getLogger("flair")


# copied from the _tie_encoder_decoder_weights method of transformers.PreTrainedModel
# but added shape check to make it safer
@staticmethod
def _tie_encoder_decoder_weights(
    encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, base_encoder_name: str
):
    uninitialized_encoder_weights: List[str] = []
    tied_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder"
            " weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        base_encoder_name: str,
        uninitialized_encoder_weights: List[str],
        depth=0,
        total_decoder_name="",
        total_encoder_name="",
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type nn.Module"
        if hasattr(decoder_pointer, "weight"):
            assert hasattr(encoder_pointer, "weight")
            # added codes start here
            if encoder_pointer.weight.shape != decoder_pointer.weight.shape:
                uninitialized_encoder_weights.append(f"{base_encoder_name}{total_encoder_name}")
                return
            # added codes end here
            encoder_pointer.weight = decoder_pointer.weight
            tied_weights.append(f"{base_encoder_name}{total_encoder_name}.weight")
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                tied_weights.append(f"{base_encoder_name}{total_encoder_name}.bias")
                encoder_pointer.bias = decoder_pointer.bias
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = {module_name + "/" + sub_name for sub_name in encoder_modules.keys()}
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is"
                        " a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    base_encoder_name,
                    uninitialized_encoder_weights,
                    depth=depth + 1,
                    total_encoder_name=f"{total_encoder_name}.{encoder_name}",
                    total_decoder_name=f"{total_decoder_name}.{decoder_name}",
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(
        decoder, encoder, base_model_prefix, base_encoder_name, uninitialized_encoder_weights
    )

    if len(uninitialized_encoder_weights) > 0:
        logger.warning(f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}")
    # added codes start here
    logger.warning(f"The following encoder weights were tied to the decoder {tied_weights}")
    # added codes end here
    return tied_weights


PreTrainedModel._tie_encoder_decoder_weights = _tie_encoder_decoder_weights


def retrieve_function_from_definition(func_definition: str) -> Callable:
    """
    Creates a function object from the definition of a function.

    Args:
        func_definition (str): the definition of a single function
    """
    local_scope = {}
    exec(func_definition, local_scope)
    parsed_ast = ast.parse(func_definition)
    function_name = [node.name for node in ast.walk(parsed_ast) if isinstance(node, ast.FunctionDef)][0]
    return local_scope[function_name]


class CausalLanguageModelDecoder(nn.Module):
    """
    A decoder module based on HuggingFace's Causal Language Models (e.g., GPT-2).
    This module is responsible for generating tokens based on the encoder's outputs.
    """

    def __init__(
        self,
        model_name: str,
        additional_special_tokens: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initializes the decoder with a pre-trained causal language model.

        Args:
            model_name (str): The name or path of the pre-trained model.
            additional_special_tokens (Dict[str, str], optional): Additional special tokens to add.
                Example: {'additional_special_tokens': '[CUSTOM]'}
        """
        super().__init__()
        self.model_name = model_name
        self.additional_special_tokens = additional_special_tokens
        self.model = AutoModelForCausalLM.from_pretrained(model_name, add_cross_attention=True, is_decoder=True)
        self.tokenizer = self._initialize_tokenizer(model_name, additional_special_tokens)

    def _initialize_tokenizer(
        self, model_name: str, additional_special_tokens: Optional[Dict[str, str]] = None
    ) -> AutoTokenizer:
        """
        Initializes the tokenizer for the decoder model.

        Args:
            model_name (str): The name or path of the pre-trained model.
            additional_special_tokens (Dict[str, str], optional): Additional special tokens to add.

        Returns:
            AutoTokenizer: The initialized tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        return self._ensure_special_tokens(tokenizer, additional_special_tokens)

    def _ensure_special_tokens(
        self,
        tokenizer: AutoTokenizer,
        additional_special_tokens: Optional[Dict[str, str]] = None,
    ) -> AutoTokenizer:
        """
        Ensures that the tokenizer has necessary special tokens.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to check.
            additional_special_tokens (Dict[str, str], optional): Additional special tokens to add.

        Returns:
            AutoTokenizer: The tokenizer with ensured special tokens.
        """
        tokens_to_check = {
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "pad_token": "[PAD]",
        }
        if additional_special_tokens:
            tokens_to_check.update(additional_special_tokens)

        special_tokens = {}
        for token_attr, token_str in tokens_to_check.items():
            # If the tokenizer doesn't have this token set, add it
            if getattr(tokenizer, token_attr) is None:
                special_tokens[token_attr] = token_str

        # If we have tokens to add, do so
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(tokenizer))
            logger.debug(f"Added special tokens {list(special_tokens.keys())} and resized decoder embeddings.")
        else:
            logger.debug("All special tokens are already present in the tokenizer.")

        return tokenizer

    def _get_state_dict(self) -> dict:
        state = {
            "model_name": self.model_name,
            "additional_special_tokens": self.additional_special_tokens,
            "model.state_dict": self.model.state_dict(),
        }
        return state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        decoder = cls(model_name=state["model_name"], additional_special_tokens=state["additional_special_tokens"])
        decoder.model.load_state_dict(state["model.state_dict"])
        logger.info("Load decoder.model from state_dict")
        return decoder


class EncoderDecoderLanguageModel(Model, GenerationMixin):
    """
    A language model based on an encoder-decoder architecture using HuggingFace's Transformers.
    """

    label_pad_token_id: int = -100  # The index to ignore when calculating cross_entropy loss

    def __init__(
        self,
        encoder_embeddings: Any,
        decoder: CausalLanguageModelDecoder,
        label_type: str,
        generate_input_text_fn: Callable[[Any], str],
        generate_output_text_fn: Callable[[Any], str],
        generate_input_text_fn_definition: Optional[str] = None,
        generate_output_text_fn_definition: Optional[str] = None,
        tie_encoder_decoder: bool = False,
    ) -> None:
        """
        Initializes the EncoderDecoderLanguageModel.

        Args:
            encoder_embeddings (Any): The embedding object (e.g., TransformerWordEmbeddings) containing the encoder.
            decoder (CausalLanguageModelDecoder): The decoder module.
            label_type (str): The type of labels (if needed for naming).
            generate_input_text_fn (Callable[[Any], str], optional):
                Callable that extracts the input text from a datapoint (could be Sentence or DataPair).
            generate_output_text_fn (Callable[[Any], str], optional):
                Callable that extracts the target text from a datapoint (could be Sentence or DataPair).
            tie_encoder_decoder (bool, optional): Whether to tie encoder and decoder weights. Defaults to False.
        """
        super().__init__()

        self.encoder_embeddings = encoder_embeddings
        self.decoder = decoder

        self._label_type = label_type

        # Store the callables
        self.generate_input_text_fn = generate_input_text_fn
        self.generate_output_text_fn = generate_output_text_fn
        self.generate_input_text_fn_definition = (
            generate_input_text_fn_definition
            if generate_input_text_fn_definition
            else inspect.getsource(generate_input_text_fn)
        )
        self.generate_output_text_fn_definition = (
            generate_output_text_fn_definition
            if generate_output_text_fn_definition
            else inspect.getsource(generate_output_text_fn)
        )

        self.tie_encoder_decoder = tie_encoder_decoder

        # Initialize EncoderDecoderModel
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            self.encoder_embeddings.model.config, self.decoder.model.config, tie_encoder_decoder=tie_encoder_decoder
        )
        self.encoder_decoder_model = EncoderDecoderModel(
            encoder=self.encoder_embeddings.model, decoder=self.decoder.model, config=config
        ).to(flair.device)
        logger.debug(f"Using Flair device: {flair.device}")
        logger.debug("EncoderDecoderModel initialized and moved to device.")

        # Initialize tokenizers
        self.encoder_tokenizer = self.encoder_embeddings.tokenizer
        self.decoder_tokenizer = self.decoder.tokenizer

        # Update key IDs in config
        self.encoder_decoder_model.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id
        self.encoder_decoder_model.config.pad_token_id = self.decoder_tokenizer.pad_token_id

    @property
    def label_type(self) -> str:
        """Returns the type of labels the model predicts."""
        return self._label_type

    def _pad_ids(self, input_ids_in_a_batch: List[List[int]], padding_value: int):
        """Pads sequences in input_ids_in_a_batch to the longest length with padding_value"""
        unpadded_input_ids = [
            torch.tensor(input_ids_in_a_sentence, dtype=torch.long).to(flair.device)
            for input_ids_in_a_sentence in input_ids_in_a_batch
        ]
        return pad_sequence(unpadded_input_ids, batch_first=True, padding_value=padding_value)

    def forward_loss(self, datapoints: List[DataPoint]) -> Tuple[torch.Tensor, int]:
        """
        Computes the forward loss for a batch of datapoints.

        Args:
            datapoints (List[DataPoint]): A batch of Flair DataPoints.

        Returns:
            Tuple[torch.Tensor, int]: The average cross entropy loss multiplied by the number of datapoints, and the number of datapoints.
        """
        if len(datapoints) == 0:
            raise RuntimeError("No datapoints provided")

        # Use the two new text-generation functions:
        input_texts = [self.generate_input_text_fn(s) for s in datapoints]
        target_texts = [self.generate_output_text_fn(s) for s in datapoints]

        encoder_inputs = self.encoder_tokenizer(
            input_texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(flair.device)

        decoder_inputs = self.decoder_tokenizer(
            text_target=target_texts,
            padding=False,
            truncation=True,
        )
        labels = self._pad_ids(decoder_inputs["input_ids"], self.label_pad_token_id)

        outputs = self.encoder_decoder_model(
            input_ids=encoder_inputs["input_ids"],
            attention_mask=encoder_inputs["attention_mask"],
            labels=labels,
            return_dict=True,
        )

        return outputs.loss * len(datapoints), len(datapoints)

    def evaluate(
        self,
        data_points: Union[List[DataPoint], torch.utils.data.dataset.Dataset],
        mini_batch_size: int = 4,
        **kwargs,
    ) -> Result:
        """
        Evaluates the model on a given dataset using cross-entropy loss and perplexity.

        Args:
            data_points (Union[List[DataPoint], torch.utils.data.dataset.Dataset]): Evaluation dataset.
            mini_batch_size (int, optional): Batch size. Defaults to 4.
            **kwargs: Additional arguments.

        Returns:
            Result: Evaluation results containing average loss and perplexity.
        """
        if not isinstance(data_points, torch.utils.data.dataset.Dataset):
            # If it's just a list, wrap it in a FlairDatapointDataset
            if isinstance(data_points, List):
                data_points = FlairDatapointDataset(data_points)
            else:
                raise ValueError("Invalid data_points type for evaluation.")

        self.encoder_decoder_model.eval()
        sum_loss = 0.0
        total_samples = 0
        data_loader = flair.datasets.DataLoader(data_points, batch_size=mini_batch_size)

        with torch.no_grad():
            for batch in data_loader:
                input_texts_batch = [self.generate_input_text_fn(dp) for dp in batch]
                target_texts_batch = [self.generate_output_text_fn(dp) for dp in batch]

                encoder_inputs_batch = self.encoder_tokenizer(
                    input_texts_batch,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                ).to(flair.device)

                decoder_inputs_batch = self.decoder_tokenizer(
                    text_target=target_texts_batch,
                    padding=False,
                    truncation=True,
                )
                labels_batch = self._pad_ids(decoder_inputs_batch["input_ids"], self.label_pad_token_id)

                outputs = self.encoder_decoder_model(
                    input_ids=encoder_inputs_batch["input_ids"],
                    attention_mask=encoder_inputs_batch["attention_mask"],
                    labels=labels_batch,
                    return_dict=True,
                )

                sum_loss += outputs.loss.item() * len(batch)
                total_samples += len(batch)

        average_loss = sum_loss / total_samples
        average_perplexity = math.exp(average_loss)

        return Result(
            main_score=average_perplexity,
            detailed_results=(
                f"Average Seq2Seq CrossEntropyLoss: {average_loss:.4f}, "
                f"Average Seq2Seq Perplexity: {average_perplexity:.4f}"
            ),
            scores={
                "loss": average_loss,
                "Seq2Seq Perplexity": average_perplexity,
            },
        )

    def predict(
        self,
        datapoints: List[DataPoint],
        decoder_input_texts: Optional[List[str]] = None,
        max_length: int = 50,
        num_beams: int = 5,
        early_stopping: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Generates predictions for a list of datapoints using the encoder-decoder model.
        Optionally allows passing initial inputs to the decoder.

        Args:
            datapoints (List[DataPoint]): List of Flair DataPoint objects or similar.
            decoder_input_texts (Optional[List[str]]): List of initial texts to pass to the decoder for each item.
            max_length (int, optional): Max length of generated sequences. Defaults to 50.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            early_stopping (bool, optional): Whether to stop beam search once at least 'num_beams' datapoints
                                             are finished. Defaults to True.
            **kwargs: Additional HF generation params.

        Returns:
            List[str]: Generated sequences.
        """
        # If user provided only a single data point, put it in a list
        if not isinstance(datapoints, list):
            datapoints = [datapoints]

        # Use generate_input_text_fn for encoder input
        input_texts = [self.generate_input_text_fn(s) for s in datapoints]

        self.encoder_decoder_model.eval()

        # Tokenize input for encoder
        encoder_inputs = self.encoder_tokenizer(
            input_texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(flair.device)

        if decoder_input_texts is not None:
            if not isinstance(decoder_input_texts, list):
                decoder_input_texts = [decoder_input_texts]
            if len(decoder_input_texts) != len(datapoints):
                raise ValueError("Length of `decoder_input_texts` must match the number of `datapoints`.")
            decoder_inputs = self.decoder_tokenizer(
                decoder_input_texts, padding="longest", truncation=True, return_tensors="pt"
            ).to(flair.device)
            generation_kwargs = {
                "decoder_input_ids": decoder_inputs["input_ids"],
                "decoder_attention_mask": decoder_inputs["attention_mask"],
            }
        else:
            generation_kwargs = {}

        # Basic generation config
        generation_kwargs.update(
            {
                "input_ids": encoder_inputs["input_ids"],
                "attention_mask": encoder_inputs["attention_mask"],
                "max_length": max_length,
                "num_beams": num_beams,
                "early_stopping": early_stopping,
                "use_cache": True,
            }
        )
        # Include additional user kwargs
        generation_kwargs.update(kwargs)

        generated_ids = self.encoder_decoder_model.generate(**generation_kwargs)
        predictions = self.decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Attach predictions as labels
        for dp, pred in zip(datapoints, predictions):
            dp.add_label(self.label_type, pred)

        return predictions

    def _get_state_dict(self) -> dict:
        state = {
            "__cls__": self.__class__.__name__,
            "encoder_embeddings": self.encoder_embeddings.save_embeddings(use_state_dict=True),
            "decoder": self.decoder._get_state_dict(),
            "label_type": self.label_type,
            "generate_input_text_fn": self.generate_input_text_fn_definition,
            "generate_output_text_fn": self.generate_output_text_fn_definition,
            "tie_encoder_decoder": self.tie_encoder_decoder,
            "encoder_decoder_model.state_dict": self.encoder_decoder_model.state_dict(),
        }
        return state

    @classmethod
    def _init_model_with_state_dict(cls, state: dict, **kwargs) -> "EncoderDecoderLanguageModel":
        encoder_embeddings = state["encoder_embeddings"]
        if isinstance(encoder_embeddings, dict):
            encoder_embeddings = load_embeddings(encoder_embeddings)
        else:
            raise NotImplementedError("Not implemented when encoder_embeddings is not a dict")

        decoder = CausalLanguageModelDecoder._init_model_with_state_dict(state["decoder"])

        generate_input_text_fn = retrieve_function_from_definition(state["generate_input_text_fn"])
        generate_output_text_fn = retrieve_function_from_definition(state["generate_output_text_fn"])

        model = cls(
            encoder_embeddings=encoder_embeddings,
            decoder=decoder,
            label_type=state["label_type"],
            generate_input_text_fn=generate_input_text_fn,
            generate_output_text_fn=generate_output_text_fn,
            generate_input_text_fn_definition=state["generate_input_text_fn"],
            generate_output_text_fn_definition=state["generate_output_text_fn"],
            tie_encoder_decoder=state["tie_encoder_decoder"],
        )

        model.encoder_decoder_model.load_state_dict(state["encoder_decoder_model.state_dict"])

        return model

    @classmethod
    def load(cls, model_path: Union[str, Path, dict[str, Any]]) -> "EncoderDecoderLanguageModel":
        from typing import cast

        return cast("EncoderDecoderLanguageModel", super().load(model_path=model_path))
