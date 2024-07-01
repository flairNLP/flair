import importlib.util
import warnings

import pytest
import torch
from PIL import Image
from transformers.utils import is_detectron2_available

from flair.data import BoundingBox, Dictionary, Sentence
from flair.embeddings import TransformerJitWordEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from tests.embedding_test_utils import BaseEmbeddingsTest


class TestTransformerWordEmbeddings(BaseEmbeddingsTest):
    embedding_cls = TransformerWordEmbeddings
    is_token_embedding = True
    is_document_embedding = False
    default_args = {"model": "distilbert-base-uncased", "allow_long_sentences": False}
    valid_args = [
        {"layers": "-1,-2,-3,-4", "layer_mean": False},
        {"layers": "all", "layer_mean": True},
        {"layers": "all", "layer_mean": False},
        {"layers": "all", "layer_mean": True, "subtoken_pooling": "mean"},
    ]

    name_field = "embeddings"
    invalid_names = ["other", "not/existing/path/to/embeddings"]

    @pytest.mark.integration()
    def test_transformer_jit_embeddings(self, results_base_path):
        base_embeddings = TransformerWordEmbeddings(
            "distilbert-base-uncased", layers="-1,-2,-3,-4", layer_mean=False, allow_long_sentences=True
        )
        sentence: Sentence = Sentence("I love Berlin, but Vienna is where my hearth is.")

        class JitWrapper(torch.nn.Module):
            def __init__(self, embedding: TransformerWordEmbeddings) -> None:
                super().__init__()
                self.embedding = embedding

            def forward(
                self,
                input_ids: torch.Tensor,
                token_lengths: torch.LongTensor,
                attention_mask: torch.Tensor,
                overflow_to_sample_mapping: torch.Tensor,
                word_ids: torch.Tensor,
            ):
                return self.embedding.forward(
                    input_ids=input_ids,
                    token_lengths=token_lengths,
                    attention_mask=attention_mask,
                    overflow_to_sample_mapping=overflow_to_sample_mapping,
                    word_ids=word_ids,
                )["token_embeddings"]

        base_embeddings.embed(sentence)
        base_token_embedding = sentence[5].get_embedding().clone()
        sentence.clear_embeddings()

        tensors = base_embeddings.prepare_tensors([sentence])
        # ensure that the prepared tensors is what we expect
        assert sorted(tensors.keys()) == [
            "attention_mask",
            "input_ids",
            "overflow_to_sample_mapping",
            "token_lengths",
            "word_ids",
        ]

        wrapper = JitWrapper(base_embeddings)
        parameter_names, parameter_list = TransformerJitWordEmbeddings.parameter_to_list(
            base_embeddings, wrapper, [sentence]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            script_module = torch.jit.trace(wrapper, parameter_list)
        jit_embeddings = TransformerJitWordEmbeddings.create_from_embedding(
            script_module, base_embeddings, parameter_names
        )

        jit_embeddings.embed(sentence)
        jit_token_embedding = sentence[5].get_embedding().clone()
        assert torch.isclose(base_token_embedding, jit_token_embedding).all()
        sentence.clear_embeddings()

        # use a SequenceTagger to save and reload the embedding in the manner it is supposed to work
        example_tagger = SequenceTagger(embeddings=jit_embeddings, tag_dictionary=Dictionary(), tag_type="none")
        results_base_path.mkdir(exist_ok=True, parents=True)
        example_tagger.save(results_base_path / "tagger.pt")
        del example_tagger
        new_example_tagger = SequenceTagger.load(results_base_path / "tagger.pt")
        loaded_jit_embedding = new_example_tagger.embeddings

        loaded_jit_embedding.embed(sentence)
        loaded_jit_token_embedding = sentence[5].get_embedding().clone()
        sentence.clear_embeddings()
        assert torch.isclose(jit_token_embedding, loaded_jit_token_embedding).all()

    def test_transformers_context_expansion(self, results_base_path):
        emb = TransformerWordEmbeddings(
            "distilbert-base-uncased", use_context=True, use_context_separator=True, respect_document_boundaries=True
        )

        # previous and next sentence as context
        sentence_previous = Sentence("How is it?")
        sentence_next = Sentence("Then again, maybe not...")

        # test expansion for sentence without context
        sentence = Sentence("This is great!")
        expanded, _ = emb._expand_sentence_with_context(sentence=sentence)
        assert " ".join([token.text for token in expanded]) == "[FLERT] This is great ! [FLERT]"

        # test expansion for with previous and next as context
        sentence = Sentence("This is great.")
        sentence._previous_sentence = sentence_previous
        sentence._next_sentence = sentence_next
        expanded, _ = emb._expand_sentence_with_context(sentence=sentence)
        assert (
            " ".join([token.text for token in expanded])
            == "How is it ? [FLERT] This is great . [FLERT] Then again , maybe not ..."
        )

        # test expansion if first sentence is document boundary
        sentence = Sentence("This is great?")
        sentence_previous.is_document_boundary = True
        sentence._previous_sentence = sentence_previous
        sentence._next_sentence = sentence_next
        expanded, _ = emb._expand_sentence_with_context(sentence=sentence)
        assert (
            " ".join([token.text for token in expanded]) == "[FLERT] This is great ? [FLERT] Then again , maybe not ..."
        )

        # test expansion if we don't use context
        emb.context_length = 0
        sentence = Sentence("I am here.")
        sentence._previous_sentence = sentence_previous
        sentence._next_sentence = sentence_next
        expanded, _ = emb._expand_sentence_with_context(sentence=sentence)
        assert " ".join([token.text for token in expanded]) == "I am here ."

    @pytest.mark.integration()
    def test_layoutlm_embeddings(self):
        sentence = Sentence(["I", "love", "Berlin"])
        sentence[0].add_metadata("bbox", BoundingBox(0, 0, 10, 10))
        sentence[1].add_metadata("bbox", (12, 0, 22, 10))
        sentence[2].add_metadata("bbox", (0, 12, 10, 22))
        emb = TransformerWordEmbeddings("microsoft/layoutlm-base-uncased", layers="-1,-2,-3,-4", layer_mean=True)
        emb.eval()
        emb.embed(sentence)

    @pytest.mark.integration()
    @pytest.mark.skipif(
        condition=not is_detectron2_available(), reason="layoutlmV2 requires detectron2 to be installed manually."
    )
    def test_layoutlmv2_embeddings(self, tasks_base_path):
        with Image.open(tasks_base_path / "example_images" / "i_love_berlin.png") as img:
            img.load()
            img = img.convert("RGB")

        sentence = Sentence(["I", "love", "Berlin"])
        sentence.add_metadata("image", img)
        sentence[0].add_metadata("bbox", BoundingBox(0, 0, 10, 10))
        sentence[1].add_metadata("bbox", (12, 0, 22, 10))
        sentence[2].add_metadata("bbox", (0, 12, 10, 22))
        emb = TransformerWordEmbeddings("microsoft/layoutlmv2-base-uncased", layers="-1,-2,-3,-4", layer_mean=True)
        emb.eval()
        emb.embed(sentence)

    @pytest.mark.integration()
    def test_layoutlmv3_embeddings(self, tasks_base_path):
        with Image.open(tasks_base_path / "example_images" / "i_love_berlin.png") as img:
            img.load()
            img = img.convert("RGB")

        sentence = Sentence(["I", "love", "Berlin"])
        sentence.add_metadata("image", img)
        sentence[0].add_metadata("bbox", BoundingBox(0, 0, 10, 10))
        sentence[1].add_metadata("bbox", (12, 0, 22, 10))
        sentence[2].add_metadata("bbox", (0, 12, 10, 22))
        emb = TransformerWordEmbeddings("microsoft/layoutlmv3-base", layers="-1,-2,-3,-4", layer_mean=True)
        emb.eval()
        emb.embed(sentence)

    @pytest.mark.integration()
    def test_layoutlmv3_embeddings_with_long_context(self, tasks_base_path):
        with Image.open(tasks_base_path / "example_images" / "i_love_berlin.png") as img:
            img.load()
            img = img.convert("RGB")

        sentence = Sentence(["I", "love", "Berlin"] * 512)
        sentence.add_metadata("image", img)
        for i in range(512):
            sentence[i * 3].add_metadata("bbox", BoundingBox(0, 0, 10, 10))
            sentence[i * 3 + 1].add_metadata("bbox", (12, 0, 22, 10))
            sentence[i * 3 + 2].add_metadata("bbox", (0, 12, 0, 10))
        emb = TransformerWordEmbeddings("microsoft/layoutlmv3-base", layers="-1,-2,-3,-4", layer_mean=True)
        emb.eval()
        emb.embed(sentence)

    @pytest.mark.integration()
    def test_ocr_embeddings_fails_when_no_bbox(self):
        sentence = Sentence(["I", "love", "Berlin"])
        emb = TransformerWordEmbeddings("microsoft/layoutlm-base-uncased", layers="-1,-2,-3,-4", layer_mean=True)
        emb.eval()
        with pytest.raises(ValueError):
            emb.embed(sentence)

    @pytest.mark.integration()
    def test_layoutlm_embeddings_with_context_warns_user(self):
        sentence = Sentence(["I", "love", "Berlin"])
        sentence[0].add_metadata("bbox", BoundingBox(0, 0, 10, 10))
        sentence[1].add_metadata("bbox", (12, 0, 22, 10))
        sentence[2].add_metadata("bbox", (0, 12, 10, 22))
        with pytest.warns(UserWarning) as record:
            TransformerWordEmbeddings("microsoft/layoutlm-base-uncased", layers="-1,-2,-3,-4", use_context=True)
        assert len(record) > 0
        assert "microsoft/layoutlm" in record[-1].message.args[0]

    @pytest.mark.integration()
    def test_layoutlmv3_without_image_embeddings_fails(self):
        sentence = Sentence(["I", "love", "Berlin"])
        sentence[0].add_metadata("bbox", BoundingBox(0, 0, 10, 10))
        sentence[1].add_metadata("bbox", (12, 0, 22, 10))
        sentence[2].add_metadata("bbox", (0, 12, 10, 22))
        emb = TransformerWordEmbeddings("microsoft/layoutlmv3-base", layers="-1,-2,-3,-4", layer_mean=True)
        emb.eval()
        with pytest.raises(ValueError):
            emb.embed(sentence)

    @pytest.mark.skipif(importlib.util.find_spec("sacremoses") is None, reason="XLM-Embeddings require 'sacremoses'")
    def test_transformer_word_embeddings_forward_language_ids(self):
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-10)
        sent_en = Sentence(["This", "is", "a", "sentence"], language_code="en")
        sent_de = Sentence(["Das", "ist", "ein", "Satz"], language_code="de")
        embeddings = TransformerWordEmbeddings("xlm-mlm-ende-1024", layers="all", allow_long_sentences=False)

        embeddings.embed([sent_de, sent_en])
        expected_similarities = [
            0.7102344036102295,
            0.7598986625671387,
            0.7437312602996826,
            0.5584433674812317,
        ]

        for token_de, token_en, exp_sim in zip(sent_de, sent_en, expected_similarities):
            sim = cos(token_de.embedding, token_en.embedding).item()
            assert abs(exp_sim - sim) < 1e-5

    def test_transformer_force_max_length(self):
        sentence: Sentence = Sentence("I love Berlin, but Vienna is where my hearth is.")
        short_embeddings = TransformerWordEmbeddings("distilbert-base-uncased", layers="-1,-2,-3,-4", layer_mean=False)
        long_embeddings = TransformerWordEmbeddings(
            "distilbert-base-uncased", layers="-1,-2,-3,-4", layer_mean=False, force_max_length=True
        )
        short_tensors = short_embeddings.prepare_tensors([sentence])
        long_tensors = long_embeddings.prepare_tensors([sentence])
        for tensor in short_tensors.values():
            if tensor.dim() > 1:  # all tensors that have a sequence length need to be shorter
                assert tensor.shape[1] < 512

        for tensor in long_tensors.values():
            if tensor.dim() > 1:  # all tensors that have a sequence length need to be exactly max length
                assert tensor.shape[1] == 512
        short_embeddings.embed(sentence)
        short_embedding_0 = sentence[0].get_embedding()
        sentence.clear_embeddings()
        long_embeddings.embed(sentence)
        long_embedding_0 = sentence[0].get_embedding()
        # apparently the precision is not that high on cuda, hence the absolute tolerance needs to be higher.
        assert torch.isclose(short_embedding_0, long_embedding_0, atol=1e-4).all()

    def test_transformers_keep_tokenizer_when_saving(self, results_base_path):
        embeddings = TransformerWordEmbeddings("distilbert-base-uncased")
        results_base_path.mkdir(exist_ok=True, parents=True)
        initial_tagger_path = results_base_path / "initial_tokenizer.pk"
        reloaded_tagger_path = results_base_path / "reloaded_tokenizer.pk"

        initial_tagger = SequenceTagger(embeddings, Dictionary(), "ner")

        initial_tagger.save(initial_tagger_path)
        reloaded_tagger = SequenceTagger.load(initial_tagger_path)

        reloaded_tagger.save(reloaded_tagger_path)

    def test_transformers_keep_tokenizer_bloom_when_saving(self, results_base_path):
        embeddings = TransformerWordEmbeddings("Muennighoff/bloom-tiny-random")
        results_base_path.mkdir(exist_ok=True, parents=True)
        initial_tagger_path = results_base_path / "initial_tokenizer.pk"
        reloaded_tagger_path = results_base_path / "reloaded_tokenizer.pk"

        initial_tagger = SequenceTagger(embeddings, Dictionary(), "ner")

        initial_tagger.save(initial_tagger_path)
        reloaded_tagger = SequenceTagger.load(initial_tagger_path)

        reloaded_tagger.save(reloaded_tagger_path)

    def test_transformer_subword_token_mapping(self):
        sentence = Sentence("El pasto es verde.")
        embeddings = TransformerWordEmbeddings("PlanTL-GOB-ES/roberta-base-biomedical-es", layers="-1")
        embeddings.embed(sentence)

    @pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None, reason="Onnx export require 'onnxruntime'")
    def test_onnx_export_works(self, results_base_path):
        texts = [
            "I live in Berlin",
            "I live in Vienna",
            "Berlin to Germany is like Vienna to Austria",
        ]

        normal_sentences = [Sentence(text) for text in texts]
        onnx_sentences = [Sentence(text) for text in texts]

        embeddings = TransformerWordEmbeddings("distilbert-base-uncased")
        results_base_path.mkdir(exist_ok=True, parents=True)
        onnx_embeddings = embeddings.export_onnx(results_base_path / "onnx-export.onnx", normal_sentences)

        embeddings.embed(normal_sentences)
        onnx_embeddings.embed(onnx_sentences)

        for sent_a, sent_b in zip(normal_sentences, onnx_sentences):
            for token_a, token_b in zip(sent_a, sent_b):
                assert torch.isclose(token_a.get_embedding(), token_b.get_embedding(), atol=1e-6).all()
