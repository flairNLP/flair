import flair
import torch
import pytest

from flair.data import Sentence
from flair.embeddings import (
    RoBERTaEmbeddings,
    OpenAIGPTEmbeddings,
    OpenAIGPT2Embeddings,
    XLNetEmbeddings,
    TransformerXLEmbeddings,
    XLMEmbeddings,
)

from pytorch_transformers import (
    RobertaModel,
    RobertaTokenizer,
    OpenAIGPTModel,
    OpenAIGPTTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    XLNetModel,
    XLNetTokenizer,
    TransfoXLModel,
    TransfoXLTokenizer,
    XLMModel,
    XLMTokenizer,
)

from typing import List


def calculate_mean_embedding(
    subword_embeddings: List[torch.FloatTensor]
) -> torch.FloatTensor:
    all_embeddings: List[torch.FloatTensor] = [
        embedding.unsqueeze(0) for embedding in subword_embeddings
    ]
    return torch.mean(torch.cat(all_embeddings, dim=0), dim=0)


@pytest.mark.slow
def test_roberta_embeddings():
    roberta_model: str = "roberta-base"

    tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
    model = RobertaModel.from_pretrained(
        pretrained_model_name_or_path=roberta_model, output_hidden_states=True
    )
    model.to(flair.device)
    model.eval()

    s: str = "Berlin and Munich have a lot of puppeteer to see ."

    with torch.no_grad():
        tokens = tokenizer.tokenize("<s> " + s + " </s>")

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(flair.device)

        hidden_states = model(tokens_tensor)[-1]

        first_layer = hidden_states[1][0]

    assert len(first_layer) == len(tokens)

    #         0           1      2       3        4         5       6      7      8       9      10     11     12     13    14      15
    #
    #       '<s>',      'Ber', 'lin', 'Ġand', 'ĠMunich', 'Ġhave', 'Ġa', 'Ġlot', 'Ġof', 'Ġpupp', 'ete', 'er', 'Ġto', 'Ġsee', 'Ġ.',  '</s>'
    #                      \     /       |        |         |       |      |      |         \      |      /     |      |      |
    #                       Berlin      and    Munich     have      a     lot     of           puppeteer        to    see     .
    #
    #                         0          1        2         3       4      5       6               7             8     9      10

    def embed_sentence(
        sentence: str,
        pooling_operation,
        layers: str = "1",
        use_scalar_mix: bool = False,
    ) -> Sentence:
        embeddings = RoBERTaEmbeddings(
            pretrained_model_name_or_path=roberta_model,
            layers=layers,
            pooling_operation=pooling_operation,
            use_scalar_mix=use_scalar_mix,
        )
        flair_sentence = Sentence(sentence)
        embeddings.embed(flair_sentence)

        return flair_sentence

    # First subword embedding
    sentence_first_subword = embed_sentence(sentence=s, pooling_operation="first")

    first_token_embedding_ref = first_layer[1].tolist()
    first_token_embedding_actual = sentence_first_subword.tokens[0].embedding.tolist()

    puppeteer_first_subword_embedding_ref = first_layer[9].tolist()
    puppeteer_first_subword_embedding_actual = sentence_first_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_first_subword_embedding_ref
        == puppeteer_first_subword_embedding_actual
    )

    # Last subword embedding
    sentence_last_subword = embed_sentence(sentence=s, pooling_operation="last")

    # First token is splitted into two subwords.
    # As we use "last" as pooling operation, we consider the last subword as "first token" here
    first_token_embedding_ref = first_layer[2].tolist()
    first_token_embedding_actual = sentence_last_subword.tokens[0].embedding.tolist()

    puppeteer_last_subword_embedding_ref = first_layer[11].tolist()
    puppeteer_last_subword_embedding_actual = sentence_last_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_last_subword_embedding_ref == puppeteer_last_subword_embedding_actual
    )

    # First and last subword embedding
    sentence_first_last_subword = embed_sentence(
        sentence=s, pooling_operation="first_last"
    )

    first_token_embedding_ref = torch.cat([first_layer[1], first_layer[2]]).tolist()
    first_token_embedding_actual = sentence_first_last_subword.tokens[
        0
    ].embedding.tolist()

    puppeteer_first_last_subword_embedding_ref = torch.cat(
        [first_layer[9], first_layer[11]]
    ).tolist()
    puppeteer_first_last_subword_embedding_actual = sentence_first_last_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_first_last_subword_embedding_ref
        == puppeteer_first_last_subword_embedding_actual
    )

    # Mean of all subword embeddings
    sentence_mean_subword = embed_sentence(sentence=s, pooling_operation="mean")

    first_token_embedding_ref = calculate_mean_embedding(
        [first_layer[1], first_layer[2]]
    ).tolist()
    first_token_embedding_actual = sentence_mean_subword.tokens[0].embedding.tolist()

    puppeteer_mean_subword_embedding_ref = calculate_mean_embedding(
        [first_layer[9], first_layer[10], first_layer[11]]
    ).tolist()
    puppeteer_mean_subword_embedding_actual = sentence_mean_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_mean_subword_embedding_ref == puppeteer_mean_subword_embedding_actual
    )

    # Check embedding dimension when using multiple layers
    sentence_mult_layers = embed_sentence(
        sentence="Munich", pooling_operation="first", layers="1,2,3,4"
    )

    ref_embedding_size = 4 * 768
    actual_embedding_size = len(sentence_mult_layers.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size

    # Check embedding dimension when using multiple layers and scalar mix
    sentence_mult_layers_scalar_mix = embed_sentence(
        sentence="Berlin",
        pooling_operation="first",
        layers="1,2,3,4",
        use_scalar_mix=True,
    )

    ref_embedding_size = 1 * 768
    actual_embedding_size = len(sentence_mult_layers_scalar_mix.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size


@pytest.mark.slow
def test_gpt_embeddings():
    gpt_model: str = "openai-gpt"

    tokenizer = OpenAIGPTTokenizer.from_pretrained(gpt_model)
    model = OpenAIGPTModel.from_pretrained(
        pretrained_model_name_or_path=gpt_model, output_hidden_states=True
    )
    model.to(flair.device)
    model.eval()

    s: str = "Berlin and Munich have a lot of puppeteer to see ."

    with torch.no_grad():
        tokens = tokenizer.tokenize(s)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(flair.device)

        hidden_states = model(tokens_tensor)[-1]

        first_layer = hidden_states[1][0]

    assert len(first_layer) == len(tokens)

    #     0             1           2            3          4         5         6        7       8       9        10        11         12
    #
    # 'berlin</w>', 'and</w>', 'munich</w>', 'have</w>', 'a</w>', 'lot</w>', 'of</w>', 'pupp', 'ete', 'er</w>', 'to</w>', 'see</w>', '.</w>'
    #     |             |           |            |          |         |         |         \      |      /          |         |          |
    #   Berlin         and        Munich        have        a        lot        of           puppeteer             to       see         .
    #
    #     0             1           2            3          4         5         6                7                  8        9          10

    def embed_sentence(
        sentence: str,
        pooling_operation,
        layers: str = "1",
        use_scalar_mix: bool = False,
    ) -> Sentence:
        embeddings = OpenAIGPTEmbeddings(
            pretrained_model_name_or_path=gpt_model,
            layers=layers,
            pooling_operation=pooling_operation,
            use_scalar_mix=use_scalar_mix,
        )
        flair_sentence = Sentence(sentence)
        embeddings.embed(flair_sentence)

        return flair_sentence

    # First subword embedding
    sentence_first_subword = embed_sentence(sentence=s, pooling_operation="first")

    first_token_embedding_ref = first_layer[0].tolist()
    first_token_embedding_actual = sentence_first_subword.tokens[0].embedding.tolist()

    puppeteer_first_subword_embedding_ref = first_layer[7].tolist()
    puppeteer_first_subword_embedding_actual = sentence_first_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_first_subword_embedding_ref
        == puppeteer_first_subword_embedding_actual
    )

    # Last subword embedding
    sentence_last_subword = embed_sentence(sentence=s, pooling_operation="last")

    first_token_embedding_ref = first_layer[0].tolist()
    first_token_embedding_actual = sentence_last_subword.tokens[0].embedding.tolist()

    puppeteer_last_subword_embedding_ref = first_layer[9].tolist()
    puppeteer_last_subword_embedding_actual = sentence_last_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_last_subword_embedding_ref == puppeteer_last_subword_embedding_actual
    )

    # First and last subword embedding
    sentence_first_last_subword = embed_sentence(
        sentence=s, pooling_operation="first_last"
    )

    first_token_embedding_ref = torch.cat([first_layer[0], first_layer[0]]).tolist()
    first_token_embedding_actual = sentence_first_last_subword.tokens[
        0
    ].embedding.tolist()

    puppeteer_first_last_subword_embedding_ref = torch.cat(
        [first_layer[7], first_layer[9]]
    ).tolist()
    puppeteer_first_last_subword_embedding_actual = sentence_first_last_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_first_last_subword_embedding_ref
        == puppeteer_first_last_subword_embedding_actual
    )

    # Mean of all subword embeddings
    sentence_mean_subword = embed_sentence(sentence=s, pooling_operation="mean")

    first_token_embedding_ref = calculate_mean_embedding([first_layer[0]]).tolist()
    first_token_embedding_actual = sentence_mean_subword.tokens[0].embedding.tolist()

    puppeteer_mean_subword_embedding_ref = calculate_mean_embedding(
        [first_layer[7], first_layer[8], first_layer[9]]
    ).tolist()
    puppeteer_mean_subword_embedding_actual = sentence_mean_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_mean_subword_embedding_ref == puppeteer_mean_subword_embedding_actual
    )

    # Check embedding dimension when using multiple layers
    sentence_mult_layers = embed_sentence(
        sentence="Munich", pooling_operation="first", layers="1,2,3,4"
    )

    ref_embedding_size = 4 * 768
    actual_embedding_size = len(sentence_mult_layers.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size

    # Check embedding dimension when using multiple layers and scalar mix
    sentence_mult_layers_scalar_mix = embed_sentence(
        sentence="Berlin",
        pooling_operation="first",
        layers="1,2,3,4",
        use_scalar_mix=True,
    )

    ref_embedding_size = 1 * 768
    actual_embedding_size = len(sentence_mult_layers_scalar_mix.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size


@pytest.mark.slow
def test_gpt2_embeddings():
    gpt_model: str = "gpt2-medium"

    tokenizer = GPT2Tokenizer.from_pretrained(gpt_model)
    model = GPT2Model.from_pretrained(
        pretrained_model_name_or_path=gpt_model, output_hidden_states=True
    )
    model.to(flair.device)
    model.eval()

    s: str = "Berlin and Munich have a lot of puppeteer to see ."

    with torch.no_grad():
        tokens = tokenizer.tokenize("<|endoftext|>" + s + "<|endoftext|>")

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(flair.device)

        hidden_states = model(tokens_tensor)[-1]

        first_layer = hidden_states[1][0]

    assert len(first_layer) == len(tokens)

    #         0           1      2       3        4         5       6      7      8       9      10     11     12     13    14          15
    #
    #  '<|endoftext|>', 'Ber', 'lin', 'Ġand', 'ĠMunich', 'Ġhave', 'Ġa', 'Ġlot', 'Ġof', 'Ġpupp', 'ete', 'er', 'Ġto', 'Ġsee', 'Ġ.', '<|endoftext|>'
    #                      \     /       |        |         |       |      |      |         \      |      /     |      |      |
    #                       Berlin      and    Munich     have      a     lot     of           puppeteer        to    see     .
    #
    #                         0          1        2         3       4      5       6               7             8     9      10

    def embed_sentence(
        sentence: str,
        pooling_operation,
        layers: str = "1",
        use_scalar_mix: bool = False,
    ) -> Sentence:
        embeddings = OpenAIGPT2Embeddings(
            pretrained_model_name_or_path=gpt_model,
            layers=layers,
            pooling_operation=pooling_operation,
            use_scalar_mix=use_scalar_mix,
        )
        flair_sentence = Sentence(sentence)
        embeddings.embed(flair_sentence)

        return flair_sentence

    # First subword embedding
    sentence_first_subword = embed_sentence(sentence=s, pooling_operation="first")

    first_token_embedding_ref = first_layer[1].tolist()
    first_token_embedding_actual = sentence_first_subword.tokens[0].embedding.tolist()

    puppeteer_first_subword_embedding_ref = first_layer[9].tolist()
    puppeteer_first_subword_embedding_actual = sentence_first_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_first_subword_embedding_ref
        == puppeteer_first_subword_embedding_actual
    )

    # Last subword embedding
    sentence_last_subword = embed_sentence(sentence=s, pooling_operation="last")

    # First token is splitted into two subwords.
    # As we use "last" as pooling operation, we consider the last subword as "first token" here
    first_token_embedding_ref = first_layer[2].tolist()
    first_token_embedding_actual = sentence_last_subword.tokens[0].embedding.tolist()

    puppeteer_last_subword_embedding_ref = first_layer[11].tolist()
    puppeteer_last_subword_embedding_actual = sentence_last_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_last_subword_embedding_ref == puppeteer_last_subword_embedding_actual
    )

    # First and last subword embedding
    sentence_first_last_subword = embed_sentence(
        sentence=s, pooling_operation="first_last"
    )

    first_token_embedding_ref = torch.cat([first_layer[1], first_layer[2]]).tolist()
    first_token_embedding_actual = sentence_first_last_subword.tokens[
        0
    ].embedding.tolist()

    puppeteer_first_last_subword_embedding_ref = torch.cat(
        [first_layer[9], first_layer[11]]
    ).tolist()
    puppeteer_first_last_subword_embedding_actual = sentence_first_last_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_first_last_subword_embedding_ref
        == puppeteer_first_last_subword_embedding_actual
    )

    # Mean of all subword embeddings
    sentence_mean_subword = embed_sentence(sentence=s, pooling_operation="mean")

    first_token_embedding_ref = calculate_mean_embedding(
        [first_layer[1], first_layer[2]]
    ).tolist()
    first_token_embedding_actual = sentence_mean_subword.tokens[0].embedding.tolist()

    puppeteer_mean_subword_embedding_ref = calculate_mean_embedding(
        [first_layer[9], first_layer[10], first_layer[11]]
    ).tolist()
    puppeteer_mean_subword_embedding_actual = sentence_mean_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_mean_subword_embedding_ref == puppeteer_mean_subword_embedding_actual
    )

    # Check embedding dimension when using multiple layers
    sentence_mult_layers = embed_sentence(
        sentence="Munich", pooling_operation="first", layers="1,2,3,4"
    )

    ref_embedding_size = 4 * 1024
    actual_embedding_size = len(sentence_mult_layers.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size

    # Check embedding dimension when using multiple layers and scalar mix
    sentence_mult_layers_scalar_mix = embed_sentence(
        sentence="Berlin",
        pooling_operation="first",
        layers="1,2,3,4",
        use_scalar_mix=True,
    )

    ref_embedding_size = 1 * 1024
    actual_embedding_size = len(sentence_mult_layers_scalar_mix.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size


@pytest.mark.slow
def test_xlnet_embeddings():
    xlnet_model: str = "xlnet-large-cased"

    tokenizer = XLNetTokenizer.from_pretrained(xlnet_model)
    model = XLNetModel.from_pretrained(
        pretrained_model_name_or_path=xlnet_model, output_hidden_states=True
    )
    model.to(flair.device)
    model.eval()

    s: str = "Berlin and Munich have a lot of puppeteer to see ."

    with torch.no_grad():
        tokens = tokenizer.tokenize("<s>" + s + "</s>")

        print(tokens)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(flair.device)

        hidden_states = model(tokens_tensor)[-1]

        first_layer = hidden_states[1][0]

    assert len(first_layer) == len(tokens)

    #   0        1         2         3         4      5      6      7        8        9      10      11   12   13     14
    #
    # '<s>', '▁Berlin', '▁and', '▁Munich', '▁have', '▁a', '▁lot', '▁of', '▁puppet', 'eer', '▁to', '▁see', '▁', '.', '</s>'
    #           |          |         |         |      |      |      |         \      /       |       |     \    /
    #         Berlin      and     Munich     have     a     lot     of       puppeteer       to     see       .
    #
    #           0          1         2         3      4      5       6           7           8       9        10

    def embed_sentence(
        sentence: str,
        pooling_operation,
        layers: str = "1",
        use_scalar_mix: bool = False,
    ) -> Sentence:
        embeddings = XLNetEmbeddings(
            pretrained_model_name_or_path=xlnet_model,
            layers=layers,
            pooling_operation=pooling_operation,
            use_scalar_mix=use_scalar_mix,
        )
        flair_sentence = Sentence(sentence)
        embeddings.embed(flair_sentence)

        return flair_sentence

    # First subword embedding
    sentence_first_subword = embed_sentence(sentence=s, pooling_operation="first")

    first_token_embedding_ref = first_layer[1].tolist()
    first_token_embedding_actual = sentence_first_subword.tokens[0].embedding.tolist()

    puppeteer_first_subword_embedding_ref = first_layer[8].tolist()
    puppeteer_first_subword_embedding_actual = sentence_first_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_first_subword_embedding_ref
        == puppeteer_first_subword_embedding_actual
    )

    # Last subword embedding
    sentence_last_subword = embed_sentence(sentence=s, pooling_operation="last")

    first_token_embedding_ref = first_layer[1].tolist()
    first_token_embedding_actual = sentence_last_subword.tokens[0].embedding.tolist()

    puppeteer_last_subword_embedding_ref = first_layer[9].tolist()
    puppeteer_last_subword_embedding_actual = sentence_last_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_last_subword_embedding_ref == puppeteer_last_subword_embedding_actual
    )

    # First and last subword embedding
    sentence_first_last_subword = embed_sentence(
        sentence=s, pooling_operation="first_last"
    )

    first_token_embedding_ref = torch.cat([first_layer[1], first_layer[1]]).tolist()
    first_token_embedding_actual = sentence_first_last_subword.tokens[
        0
    ].embedding.tolist()

    puppeteer_first_last_subword_embedding_ref = torch.cat(
        [first_layer[8], first_layer[9]]
    ).tolist()
    puppeteer_first_last_subword_embedding_actual = sentence_first_last_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_first_last_subword_embedding_ref
        == puppeteer_first_last_subword_embedding_actual
    )

    # Mean of all subword embeddings
    sentence_mean_subword = embed_sentence(sentence=s, pooling_operation="mean")

    first_token_embedding_ref = calculate_mean_embedding([first_layer[1]]).tolist()
    first_token_embedding_actual = sentence_mean_subword.tokens[0].embedding.tolist()

    puppeteer_mean_subword_embedding_ref = calculate_mean_embedding(
        [first_layer[8], first_layer[9]]
    ).tolist()
    puppeteer_mean_subword_embedding_actual = sentence_mean_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_mean_subword_embedding_ref == puppeteer_mean_subword_embedding_actual
    )

    # Check embedding dimension when using multiple layers
    sentence_mult_layers = embed_sentence(
        sentence="Munich", pooling_operation="first", layers="1,2,3,4"
    )

    ref_embedding_size = 4 * model.d_model
    actual_embedding_size = len(sentence_mult_layers.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size

    # Check embedding dimension when using multiple layers and scalar mix
    sentence_mult_layers_scalar_mix = embed_sentence(
        sentence="Berlin",
        pooling_operation="first",
        layers="1,2,3,4",
        use_scalar_mix=True,
    )

    ref_embedding_size = 1 * model.d_model
    actual_embedding_size = len(sentence_mult_layers_scalar_mix.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size


@pytest.mark.slow
def test_transformer_xl_embeddings():
    transfo_model: str = "transfo-xl-wt103"

    tokenizer = TransfoXLTokenizer.from_pretrained(transfo_model)
    model = TransfoXLModel.from_pretrained(
        pretrained_model_name_or_path=transfo_model, output_hidden_states=True
    )
    model.to(flair.device)
    model.eval()

    s: str = "Berlin and Munich have a lot of puppeteer to see ."

    with torch.no_grad():
        tokens = tokenizer.tokenize(s + "<eos>")

        print(tokens)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(flair.device)

        hidden_states = model(tokens_tensor)[-1]

        first_layer = hidden_states[1][0]

    assert len(first_layer) == len(tokens)

    #     0       1        2        3     4     5      6        7        8      9     10     11
    #
    # 'Berlin', 'and', 'Munich', 'have', 'a', 'lot', 'of', 'puppeteer', 'to', 'see', '.', '<eos>'
    #     |       |        |        |     |     |      |        |        |      |     |
    #  Berlin    and    Munich    have    a    lot    of    puppeteer    to    see    .
    #
    #     0       1        2        3     4     5      6        7        8      9     10

    def embed_sentence(
        sentence: str, layers: str = "1", use_scalar_mix: bool = False
    ) -> Sentence:
        embeddings = TransformerXLEmbeddings(
            pretrained_model_name_or_path=transfo_model,
            layers=layers,
            use_scalar_mix=use_scalar_mix,
        )
        flair_sentence = Sentence(sentence)
        embeddings.embed(flair_sentence)

        return flair_sentence

    sentence = embed_sentence(sentence=s)

    first_token_embedding_ref = first_layer[0].tolist()
    first_token_embedding_actual = sentence.tokens[0].embedding.tolist()

    puppeteer_embedding_ref = first_layer[7].tolist()
    puppeteer_embedding_actual = sentence.tokens[7].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert puppeteer_embedding_ref == puppeteer_embedding_actual

    # Check embedding dimension when using multiple layers
    sentence_mult_layers = embed_sentence(sentence="Munich", layers="1,2,3,4")

    ref_embedding_size = 4 * model.d_embed
    actual_embedding_size = len(sentence_mult_layers.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size

    # Check embedding dimension when using multiple layers and scalar mix
    sentence_mult_layers_scalar_mix = embed_sentence(
        sentence="Berlin", layers="1,2,3,4", use_scalar_mix=True
    )

    ref_embedding_size = 1 * model.d_embed
    actual_embedding_size = len(sentence_mult_layers_scalar_mix.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size


@pytest.mark.slow
def test_xlm_embeddings():
    xlm_model: str = "xlm-mlm-en-2048"

    tokenizer = XLMTokenizer.from_pretrained(xlm_model)
    model = XLMModel.from_pretrained(
        pretrained_model_name_or_path=xlm_model, output_hidden_states=True
    )
    model.to(flair.device)
    model.eval()

    s: str = "Berlin and Munich have a lot of puppeteer to see ."

    with torch.no_grad():
        tokens = tokenizer.tokenize("<s>" + s + "</s>")

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(flair.device)

        hidden_states = model(tokens_tensor)[-1]

        first_layer = hidden_states[1][0]

    assert len(first_layer) == len(tokens)

    #    0      1             2           3            4          5         6         7         8       9      10        11       12         13        14
    #
    #   <s>  'berlin</w>', 'and</w>', 'munich</w>', 'have</w>', 'a</w>', 'lot</w>', 'of</w>', 'pupp', 'ete', 'er</w>', 'to</w>', 'see</w>', '.</w>', '</s>
    #           |             |           |            |          |         |         |         \      |      /          |         |          |
    #         Berlin         and        Munich        have        a        lot        of           puppeteer             to       see         .
    #
    #           0             1           2            3          4         5          6               7                  8        9          10

    def embed_sentence(
        sentence: str,
        pooling_operation,
        layers: str = "1",
        use_scalar_mix: bool = False,
    ) -> Sentence:
        embeddings = XLMEmbeddings(
            pretrained_model_name_or_path=xlm_model,
            layers=layers,
            pooling_operation=pooling_operation,
            use_scalar_mix=use_scalar_mix,
        )
        flair_sentence = Sentence(sentence)
        embeddings.embed(flair_sentence)

        return flair_sentence

    # First subword embedding
    sentence_first_subword = embed_sentence(sentence=s, pooling_operation="first")

    first_token_embedding_ref = first_layer[1].tolist()
    first_token_embedding_actual = sentence_first_subword.tokens[0].embedding.tolist()

    puppeteer_first_subword_embedding_ref = first_layer[8].tolist()
    puppeteer_first_subword_embedding_actual = sentence_first_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_first_subword_embedding_ref
        == puppeteer_first_subword_embedding_actual
    )

    # Last subword embedding
    sentence_last_subword = embed_sentence(sentence=s, pooling_operation="last")

    first_token_embedding_ref = first_layer[1].tolist()
    first_token_embedding_actual = sentence_last_subword.tokens[0].embedding.tolist()

    puppeteer_last_subword_embedding_ref = first_layer[10].tolist()
    puppeteer_last_subword_embedding_actual = sentence_last_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_last_subword_embedding_ref == puppeteer_last_subword_embedding_actual
    )

    # First and last subword embedding
    sentence_first_last_subword = embed_sentence(
        sentence=s, pooling_operation="first_last"
    )

    first_token_embedding_ref = torch.cat([first_layer[1], first_layer[1]]).tolist()
    first_token_embedding_actual = sentence_first_last_subword.tokens[
        0
    ].embedding.tolist()

    puppeteer_first_last_subword_embedding_ref = torch.cat(
        [first_layer[8], first_layer[10]]
    ).tolist()
    puppeteer_first_last_subword_embedding_actual = sentence_first_last_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_first_last_subword_embedding_ref
        == puppeteer_first_last_subword_embedding_actual
    )

    # Mean of all subword embeddings
    sentence_mean_subword = embed_sentence(sentence=s, pooling_operation="mean")

    first_token_embedding_ref = calculate_mean_embedding([first_layer[1]]).tolist()
    first_token_embedding_actual = sentence_mean_subword.tokens[0].embedding.tolist()

    puppeteer_mean_subword_embedding_ref = calculate_mean_embedding(
        [first_layer[8], first_layer[9], first_layer[10]]
    ).tolist()
    puppeteer_mean_subword_embedding_actual = sentence_mean_subword.tokens[
        7
    ].embedding.tolist()

    assert first_token_embedding_ref == first_token_embedding_actual
    assert (
        puppeteer_mean_subword_embedding_ref == puppeteer_mean_subword_embedding_actual
    )

    # Check embedding dimension when using multiple layers
    sentence_mult_layers = embed_sentence(
        sentence="Munich", pooling_operation="first", layers="1,2,3,4"
    )

    ref_embedding_size = 4 * model.embeddings.embedding_dim
    actual_embedding_size = len(sentence_mult_layers.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size

    # Check embedding dimension when using multiple layers and scalar mix
    sentence_mult_layers_scalar_mix = embed_sentence(
        sentence="Berlin",
        pooling_operation="first",
        layers="1,2,3,4",
        use_scalar_mix=True,
    )

    ref_embedding_size = 1 * model.embeddings.embedding_dim
    actual_embedding_size = len(sentence_mult_layers_scalar_mix.tokens[0].embedding)

    assert ref_embedding_size == actual_embedding_size
