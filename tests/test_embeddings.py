import importlib.util
import warnings

import pytest
import torch

from flair.data import Dictionary, Sentence
from flair.embeddings import (
    DocumentCNNEmbeddings,
    DocumentLMEmbeddings,
    DocumentPoolEmbeddings,
    DocumentRNNEmbeddings,
    FlairEmbeddings,
    StackedEmbeddings,
    TokenEmbeddings,
    TransformerDocumentEmbeddings,
    TransformerWordEmbeddings,
    WordEmbeddings,
)
from flair.embeddings.transformer import TransformerJitWordEmbeddings
from flair.models import LanguageModel, SequenceTagger

glove: TokenEmbeddings = WordEmbeddings("turian")
flair_embedding: TokenEmbeddings = FlairEmbeddings("news-forward-fast")


def test_load_non_existing_embedding():
    with pytest.raises(ValueError):
        WordEmbeddings("other")

    with pytest.raises(ValueError):
        WordEmbeddings("not/existing/path/to/embeddings")


def test_load_non_existing_flair_embedding():
    with pytest.raises(ValueError):
        FlairEmbeddings("other")


def test_keep_batch_order():
    embeddings = DocumentRNNEmbeddings([glove])
    sentences_1 = [Sentence("First sentence"), Sentence("This is second sentence")]
    sentences_2 = [Sentence("This is second sentence"), Sentence("First sentence")]

    embeddings.embed(sentences_1)
    embeddings.embed(sentences_2)

    assert sentences_1[0].to_original_text() == "First sentence"
    assert sentences_1[1].to_original_text() == "This is second sentence"

    assert torch.norm(sentences_1[0].embedding - sentences_2[1].embedding) == 0.0
    assert torch.norm(sentences_1[0].embedding - sentences_2[1].embedding) == 0.0
    del embeddings


def test_stacked_embeddings():
    embeddings: StackedEmbeddings = StackedEmbeddings([glove, flair_embedding])

    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")
    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) == 1074

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0
    del embeddings


def test_transformer_word_embeddings_also_set_trainling_whitespaces():
    embeddings = TransformerWordEmbeddings("distilbert-base-uncased")

    sentence: Sentence = Sentence(["hello", " ", "hm", " "])
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) > 0


def test_transformer_word_embeddings():
    embeddings = TransformerWordEmbeddings("distilbert-base-uncased", layers="-1,-2,-3,-4", layer_mean=False)

    sentence: Sentence = Sentence("I love Berlin")
    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) == 3072

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0

    embeddings = TransformerWordEmbeddings("distilbert-base-uncased", layers="all", layer_mean=False)

    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) == 5376

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0
    del embeddings

    embeddings = TransformerWordEmbeddings("distilbert-base-uncased", layers="all", layer_mean=True)

    embeddings.embed(sentence)

    for token in sentence.tokens:
        assert len(token.get_embedding()) == 768

        token.clear_embeddings()

        assert len(token.get_embedding()) == 0
    del embeddings


@pytest.mark.skipif(importlib.util.find_spec("sacremoses") is None, reason="XLM-Embeddings require 'sacremoses'")
def test_transformer_word_embeddings_forward_language_ids():
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

    for (token_de, token_en, exp_sim) in zip(sent_de, sent_en, expected_similarities):
        sim = cos(token_de.embedding, token_en.embedding).item()
        assert abs(exp_sim - sim) < 1e-5


@pytest.mark.integration
def test_transformer_jit_embeddings(results_base_path):
    base_embeddings = TransformerWordEmbeddings(
        "distilbert-base-uncased", layers="-1,-2,-3,-4", layer_mean=False, allow_long_sentences=True
    )
    sentence: Sentence = Sentence("I love Berlin, but Vienna is where my hearth is.")

    class JitWrapper(torch.nn.Module):
        def __init__(self, embedding: TransformerWordEmbeddings):
            super().__init__()
            self.embedding = embedding

        def forward(
            self,
            input_ids: torch.Tensor,
            lengths: torch.LongTensor,
            attention_mask: torch.Tensor,
            overflow_to_sample_mapping: torch.Tensor,
            word_ids: torch.Tensor,
        ):
            return self.embedding.forward(
                input_ids=input_ids,
                lengths=lengths,
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
        "lengths",
        "overflow_to_sample_mapping",
        "word_ids",
    ]

    wrapper = JitWrapper(base_embeddings)
    parameter_names, parameter_list = TransformerJitWordEmbeddings.parameter_to_list(
        base_embeddings, wrapper, [sentence]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        script_module = torch.jit.trace(wrapper, parameter_list)
    jit_embeddings = TransformerJitWordEmbeddings.create_from_embedding(script_module, base_embeddings, parameter_names)

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


def test_transformer_force_max_length():
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


def test_transformer_weird_sentences():
    embeddings = TransformerWordEmbeddings("distilbert-base-uncased", layers="all", layer_mean=True)

    sentence = Sentence("Hybrid mesons , qq ̄ states with an admixture")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("typical proportionalities of ∼ 1nmV − 1 [ 3,4 ] .")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("🤟 🤟  🤟 hüllo")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("🤟hallo 🤟 🤟 🤟 🤟")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("🤟hallo 🤟 🤟 🤟 🤟")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("🤟")
    embeddings.embed(sentence)
    for token in sentence:
        assert len(token.get_embedding()) == 768

    sentence = Sentence("🤟")
    sentence_2 = Sentence("second sentence")
    embeddings.embed([sentence, sentence_2])
    for token in sentence:
        assert len(token.get_embedding()) == 768
    for token in sentence_2:
        assert len(token.get_embedding()) == 768


def test_fine_tunable_flair_embedding():
    language_model_forward = LanguageModel(Dictionary.load("chars"), is_forward_lm=True, hidden_size=32, nlayers=1)

    embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
        [FlairEmbeddings(language_model_forward, fine_tune=True)],
        hidden_size=128,
        bidirectional=False,
    )

    sentence: Sentence = Sentence("I love Berlin.")

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 128
    assert len(sentence.get_embedding()) == embeddings.embedding_length

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0

    embeddings: DocumentLMEmbeddings = DocumentLMEmbeddings([FlairEmbeddings(language_model_forward, fine_tune=True)])

    sentence: Sentence = Sentence("I love Berlin.")

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 32
    assert len(sentence.get_embedding()) == embeddings.embedding_length

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0
    del embeddings


def test_document_lstm_embeddings():
    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")

    embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
        [glove, flair_embedding], hidden_size=128, bidirectional=False
    )

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 128
    assert len(sentence.get_embedding()) == embeddings.embedding_length

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0
    del embeddings


def test_document_bidirectional_lstm_embeddings():
    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")

    embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
        [glove, flair_embedding], hidden_size=128, bidirectional=True
    )

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 512
    assert len(sentence.get_embedding()) == embeddings.embedding_length

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0
    del embeddings


def test_document_pool_embeddings():
    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")

    for mode in ["mean", "max", "min"]:
        embeddings: DocumentPoolEmbeddings = DocumentPoolEmbeddings(
            [glove, flair_embedding], pooling=mode, fine_tune_mode="none"
        )

        embeddings.embed(sentence)

        assert len(sentence.get_embedding()) == 1074

        sentence.clear_embeddings()

        assert len(sentence.get_embedding()) == 0
        del embeddings


def test_document_pool_embeddings_nonlinear():
    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")

    for mode in ["mean", "max", "min"]:
        embeddings: DocumentPoolEmbeddings = DocumentPoolEmbeddings(
            [glove, flair_embedding], pooling=mode, fine_tune_mode="nonlinear"
        )

        embeddings.embed(sentence)

        assert len(sentence.get_embedding()) == 1074

        sentence.clear_embeddings()

        assert len(sentence.get_embedding()) == 0
        del embeddings


def test_transformer_document_embeddings():
    embeddings = TransformerDocumentEmbeddings("distilbert-base-uncased")

    sentence: Sentence = Sentence("I love Berlin")
    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 768

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0

    embeddings = TransformerDocumentEmbeddings("distilbert-base-uncased", layers="all")

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 5376

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0

    embeddings = TransformerDocumentEmbeddings("distilbert-base-uncased", layers="all", layer_mean=True)

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 768

    sentence.clear_embeddings()

    del embeddings


def test_document_cnn_embeddings():
    sentence: Sentence = Sentence("I love Berlin. Berlin is a great place to live.")

    embeddings: DocumentCNNEmbeddings = DocumentCNNEmbeddings([glove, flair_embedding], kernels=((50, 2), (50, 3)))

    embeddings.embed(sentence)

    assert len(sentence.get_embedding()) == 100
    assert len(sentence.get_embedding()) == embeddings.embedding_length

    sentence.clear_embeddings()

    assert len(sentence.get_embedding()) == 0
    del embeddings


def test_transformers_keep_tokenizer_when_saving(results_base_path):
    embeddings = TransformerWordEmbeddings("sentence-transformers/paraphrase-albert-small-v2")
    results_base_path.mkdir(exist_ok=True, parents=True)
    initial_tagger_path = results_base_path / "initial_tokenizer.pk"
    reloaded_tagger_path = results_base_path / "reloaded_tokenizer.pk"

    initial_tagger = SequenceTagger(embeddings, Dictionary(), "ner")

    initial_tagger.save(initial_tagger_path)
    reloaded_tagger = SequenceTagger.load(initial_tagger_path)

    reloaded_tagger.save(reloaded_tagger_path)


def test_transformers_keep_tokenizer_bloom_when_saving(results_base_path):
    embeddings = TransformerWordEmbeddings("Muennighoff/bloom-tiny-random")
    results_base_path.mkdir(exist_ok=True, parents=True)
    initial_tagger_path = results_base_path / "initial_tokenizer.pk"
    reloaded_tagger_path = results_base_path / "reloaded_tokenizer.pk"

    initial_tagger = SequenceTagger(embeddings, Dictionary(), "ner")

    initial_tagger.save(initial_tagger_path)
    reloaded_tagger = SequenceTagger.load(initial_tagger_path)

    reloaded_tagger.save(reloaded_tagger_path)


def test_transformer_subword_token_mapping():
    sentence = Sentence("El pasto es verde.")
    embeddings = TransformerWordEmbeddings("PlanTL-GOB-ES/roberta-base-biomedical-es", layers="-1")
    embeddings.embed(sentence)
